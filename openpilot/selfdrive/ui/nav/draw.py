"""Drawing for the navigation views: thick polylines for the map, and the card's icons, lane arrows and fades, as vector
strokes so they scale to any layout."""
import math
from functools import lru_cache

import numpy as np
import pyray as rl

MITER_LIMIT = 0.5  # cos of half the turn at a joint, past which its miter is cut short


def miter_normals(points: np.ndarray) -> np.ndarray:
  """Each point's offset for a stroke of half-width 1: the segments' normal at the ends, the mitre at joints."""
  d = np.diff(points, axis=0)
  n = np.stack([-d[:, 1], d[:, 0]], axis=1) / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)[:, None]
  normals = np.empty((len(points), 2))
  normals[0], normals[-1] = n[0], n[-1]
  m = n[:-1] + n[1:]
  m /= np.maximum(np.hypot(m[:, 0], m[:, 1]), 1e-6)[:, None]
  normals[1:-1] = m / np.maximum((m * n[1:]).sum(axis=1), MITER_LIMIT)[:, None]
  return normals


def strip(points: np.ndarray, half: float) -> np.ndarray:
  """A polyline's outline (screen points) as a triangle strip `half` either side, mitred at its joints."""
  normals = miter_normals(points)
  out = np.empty((2 * len(points), 2), np.float32)
  out[0::2] = points - normals * half  # this winding faces the camera: raylib culls the other
  out[1::2] = points + normals * half
  return out


class StrokeBatch:
  """Many polylines of one colour, each of its own width in px, mitred and square-capped. They're laid out in their own
  coordinates (metres east and north, y up) for a scale, and drawn under a transform onto the screen (a rotation, the
  scale and the flip of y) as a few strips joined by degenerate triangles, so a frame does no work on them in Python:
  they're laid out again only when the scale has moved enough to change their widths."""
  CHUNK = 6000  # strip vertices per draw call, within raylib's batch
  RESCALE = 0.02  # the scale's change, as a fraction, that lays them out again (their widths off by up to this)

  def __init__(self, lines: list[np.ndarray], widths: list[float]):
    anchors, offsets, halves, chunks, chunk, size = [], [], [], [], [], 0
    base = 0
    for pts, width in zip(lines, widths, strict=True):
      pts = np.asarray(pts, np.float64)
      if len(pts) >= 2:
        pts = pts[np.concatenate(([True], np.hypot(*np.diff(pts, axis=0).T) > 0.1))]
      if len(pts) < 2:
        continue
      n = miter_normals(pts)
      off = np.empty((2 * len(pts), 2))
      off[0::2], off[1::2] = n, -n  # y up: the flip onto the screen turns this into the culled-safe winding
      d0, d1 = pts[1] - pts[0], pts[-1] - pts[-2]
      off[:2] -= d0 / max(float(np.hypot(*d0)), 1e-6)  # square caps, over the gaps where roads meet
      off[-2:] += d1 / max(float(np.hypot(*d1)), 1e-6)
      anchors.append(np.repeat(pts, 2, axis=0))
      offsets.append(off)
      halves.append(np.full(2 * len(pts), width / 2))
      k = 2 * len(pts)
      if chunk and size + k + 2 > self.CHUNK:
        chunks.append(np.concatenate(chunk))
        chunk, size = [], 0
      if chunk:  # a degenerate join: the last vertex again, then the next's first; an even count keeps the winding
        chunk.append(np.array([chunk[-1][-1], base]))
        size += 2
      chunk.append(np.arange(base, base + k))
      size += k
      base += k
    if chunk:
      chunks.append(np.concatenate(chunk))
    self.anchors = np.concatenate(anchors) if anchors else np.zeros((0, 2))
    self.offsets = np.concatenate(offsets) if offsets else np.zeros((0, 2))
    self.halves = np.concatenate(halves)[:, None] if halves else np.zeros((0, 1))
    self.chunks = chunks
    self._scale: float | None = None
    self._strips: list = []  # (vertices, their buffer) per chunk, at self._scale

  def draw(self, origin: np.ndarray, centre: np.ndarray, angle: float, scale: float, color: rl.Color) -> None:
    """Draws with `centre` (in the lines' coordinates) at screen point `origin`, turned `angle` degrees
    counterclockwise and scaled px per unit."""
    if not self.chunks:
      return
    if self._scale is None or abs(scale / self._scale - 1.0) > self.RESCALE:
      self._scale = scale
      verts = (self.anchors + self.offsets * self.halves / scale).astype(np.float32)
      self._strips = []
      for idx in self.chunks:
        s = np.ascontiguousarray(verts[idx])
        self._strips.append((s, rl.ffi.from_buffer("Vector2[]", s)))
    rl.rl_push_matrix()
    rl.rl_translatef(float(origin[0]), float(origin[1]), 0.0)
    rl.rl_scalef(scale, -scale, 1.0)
    rl.rl_rotatef(angle, 0.0, 0.0, 1.0)
    rl.rl_translatef(-float(centre[0]), -float(centre[1]), 0.0)
    for s, buf in self._strips:
      rl.draw_triangle_strip(buf, len(s), color)
    rl.rl_pop_matrix()


def draw_polyline(points: np.ndarray, thickness: float, color: rl.Color, round_caps: bool = False) -> None:
  """One draw call per polyline, from an (n, 2) array of screen points."""
  pts = np.asarray(points, np.float32)
  if len(pts) >= 2:
    keep = np.concatenate(([True], np.hypot(*np.diff(pts, axis=0).T) > 0.5))
    pts = pts[keep]
  if len(pts) < 2:
    return
  s = np.ascontiguousarray(strip(pts, thickness / 2))
  rl.draw_triangle_strip(rl.ffi.from_buffer("Vector2[]", s), len(s), color)
  if round_caps:
    for p in (pts[0], pts[-1]):
      rl.draw_circle_v(rl.Vector2(float(p[0]), float(p[1])), thickness / 2, color)


def lerp(a, b, t):
  return a + (b - a) * t


def clamp01(t: float) -> float:
  return min(max(t, 0.0), 1.0)


def ease(t: float) -> float:
  """Ease-out cubic."""
  return 1 - (1 - clamp01(t)) ** 3


def smoothstep(t: float) -> float:
  t = clamp01(t)
  return t * t * (3 - 2 * t)


def smootherstep(t: float) -> float:
  t = clamp01(t)
  return t * t * t * (t * (6 * t - 15) + 10)


class Anim:
  """A value easing towards its target over `dur` seconds."""
  def __init__(self, v: float):
    self.a = self.b = v
    self.t0 = -1e9
    self.dur = 0.7

  def set(self, target: float, now: float, dur: float):
    if target != self.b:
      self.a, self.b, self.t0, self.dur = self.value(now), target, now, dur

  def value(self, now: float) -> float:
    return lerp(self.a, self.b, ease((now - self.t0) / self.dur))

  def hold_at(self, v: float, now: float):
    """Restart from v (where a drag left it) towards the current target."""
    self.a, self.t0 = v, now

  def linear(self, now: float) -> float:
    """The move on plain time, not eased: for crossfades, which an ease-out would rush through in a frame or two."""
    return lerp(self.a, self.b, clamp01((now - self.t0) / self.dur))

  def fade(self, now: float) -> float:
    return lerp(self.a, self.b, clamp01((now - self.t0) / (0.6 * self.dur)))


def xfade(t: float) -> tuple[float, float]:
  """(outgoing, incoming) alphas for swapping one icon for another over t 0..1: the old one is mostly gone before the
  new one shows, so they never sit half on top of each other."""
  return clamp01(1 - t / 0.55), clamp01((t - 0.4) / 0.6)


def rgba(c) -> tuple[int, int, int, int]:
  """A colour's channels, from an rl.Color or a tuple (as pyray's own colour constants are)."""
  return (c.r, c.g, c.b, c.a) if hasattr(c, "r") else tuple(c)


def mix(a, b, t: float) -> rl.Color:
  (ar, ag, ab, aa), (br, bg, bb, ba) = rgba(a), rgba(b)
  return rl.Color(int(lerp(ar, br, t)), int(lerp(ag, bg, t)), int(lerp(ab, bb, t)), int(lerp(aa, ba, t)))


def with_alpha(c, a: float) -> rl.Color:
  r, g, b, ca = rgba(c)
  return rl.Color(r, g, b, int(ca * clamp01(a)))


def premul(c, a: float) -> rl.Color:
  """c faded toward black by a, opaque: for drawing under max_blend."""
  r, g, b, _ = rgba(c)
  return rl.Color(int(r * a), int(g * a), int(b * a), 255)


GL_ONE, GL_MAX = 1, 0x8008


class max_blend:
  """Brightest wins: overlapping strokes drawn premultiplied (premul) fade as one, without doubling up where they
  cross, and on a near-black ground that reads as a fade. Render textures would do it too but broke on the 3X (nothing
  drawn, flicker), so don't use them mid-frame there."""
  def __enter__(self):
    rl.rl_set_blend_factors(GL_ONE, GL_ONE, GL_MAX)
    rl.begin_blend_mode(rl.BlendMode.BLEND_CUSTOM)

  def __exit__(self, *_):
    rl.end_blend_mode()


def arc(cx: float, cy: float, r: float, a0: float, a1: float, n: int = 10) -> list[tuple[float, float]]:
  """n + 1 points on a circle (screen angles in degrees, y down) from a0 to a1."""
  return [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)), cy + r * math.sin(math.radians(a0 + (a1 - a0) * i / n)))
          for i in range(n + 1)]


# The card's icons are built once per place, size and animation step and drawn from that while they stay put: the
# geometry functions are cached on their arguments rounded to a quarter pixel (and steps to a hundredth)
def q(v: float) -> float:
  return round(v * 4) / 4


class Shape:
  """Pieces of one colour drawn in one call: each piece a triangle strip of its own (a line's quad, a round join's
  polygon zigzagged across, a lone triangle), chained into one strip by repeating the vertices either side of each
  join (degenerate triangles, which draw nothing). raylib's strip alternates each triangle's vertex order to keep the
  winding, so each piece starts on whichever parity puts its triangles screen-counterclockwise (the side raylib
  doesn't cull)."""
  def __init__(self, pieces):
    seq: list = []
    for p in pieces:
      if len(p) < 3:
        continue
      (ax, ay), (bx, by), (cx, cy) = p[0], p[1], p[2]
      parity = 0 if (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) <= 0 else 1  # y down: <= 0 is counterclockwise
      if seq:
        seq += [seq[-1], p[0]]
      if len(seq) % 2 != parity:
        seq.append(p[0])
      seq += p
    self.n = len(seq)
    if self.n:
      self._points = np.ascontiguousarray(seq, np.float32)
      self._buf = rl.ffi.from_buffer("Vector2[]", self._points)

  def draw(self, color: rl.Color) -> None:
    if self.n:
      rl.draw_triangle_strip(self._buf, self.n, color)


def stroke_tris(paths, w: float) -> list:
  """Strokes through each path's points, w wide, round at their joints and ends, as strip pieces for a Shape."""
  pieces = []
  r = w / 2
  k = max(8, min(24, int(w * 2)))  # as few segments as still look round at the width
  ring = [(math.cos(2 * math.pi * i / k) * r, math.sin(2 * math.pi * i / k) * r) for i in range(k)]
  zigzag = [0] + [j for i in range(1, k // 2 + 1) for j in (i, k - i)][:k - 1]  # the polygon as a strip, across
  for path in paths:
    for (ax, ay), (bx, by) in zip(path, path[1:], strict=False):
      d = math.hypot(bx - ax, by - ay)
      if d < 1e-6:
        continue
      nx, ny = -(by - ay) / d * r, (bx - ax) / d * r
      pieces.append([(ax + nx, ay + ny), (ax - nx, ay - ny), (bx + nx, by + ny), (bx - nx, by - ny)])
    for px, py in path:
      pieces.append([(px + ring[i][0], py + ring[i][1]) for i in zigzag])
  return pieces


def arrow_tris(tip, l, notch, r) -> list:
  return [[tip, l, notch], [tip, notch, r]]


Strokes = list[list[tuple[float, float]]]


def maneuver_kind(kind: str, modifier: str) -> tuple[str, bool]:
  """The turn icon for a maneuver (OSRM type and modifier) and whether it's mirrored for a left one."""
  left = "left" in modifier
  if kind == "arrive":
    return "arrive", False
  if kind in ("roundabout", "exit roundabout"):
    return "roundabout", False
  if modifier == "uturn":
    return "uturn", False
  if kind == "merge":
    return ("merge", False) if modifier == "straight" else ("slight", left)
  if kind == "off ramp" and modifier != "straight":
    return "exit", left
  if modifier.startswith("sharp"):
    return "sharp", left
  if modifier.startswith("slight") or kind in ("fork", "on ramp") and modifier != "straight":
    return "slight", left
  if modifier in ("left", "right"):
    return "right", left
  return "straight", False


def maneuver_paths(kind: str) -> Strokes:
  """The turn icon as strokes in a 48-unit box (y down), for a right-hand maneuver."""
  if kind == "right":
    return [[(14, 42), (14, 24)] + arc(22, 24, 8, 180, 270) + [(34, 16)], [(28, 9), (35, 16), (28, 23)]]
  if kind == "slight":
    return [[(18, 42), (18, 28), (30, 13)], [(22, 11), (31, 12), (31, 21)]]
  if kind == "exit":
    return [[(18, 44), (18, 26)] + [(18 + 14 * (1 - math.cos(t * math.pi / 2)), 26 - 17 * math.sin(t * math.pi / 2))
                                    for t in [i / 8 for i in range(1, 9)]], [(25, 6), (33, 8.5), (31, 16.5)]]
  if kind == "uturn":
    return [[(32, 42), (32, 20)] + arc(24, 20, 8, 0, -180) + [(16, 30)], [(10, 25), (16, 31), (22, 25)]]
  if kind == "sharp":
    return [[(16, 43), (16, 12), (34, 32)], [(34, 21), (34, 32), (23, 32)]]
  if kind == "merge":
    return [[(14, 43), (14, 34), (24, 24)], [(34, 43), (34, 34), (24, 24), (24, 8)], [(14, 18), (24, 8), (34, 18)]]
  if kind == "roundabout":  # in at the bottom, round the right side as traffic on the right goes, and out at the top
    return [[(18, 45), (18, 37)] + arc(18, 28, 9, 90, -90, 14) + [(18, 5)], [(10, 13), (18, 5), (26, 13)]]
  if kind == "arrive":  # a map pin
    return [arc(24, 18, 9, 0, 360, 20), [(24, 27), (24, 42)]]
  return [[(24, 42), (24, 8)], [(14, 18), (24, 8), (34, 18)]]


@lru_cache(maxsize=32)
def _maneuver(kind: str, left: bool, cx: float, cy: float, size: float) -> Shape:
  s = size / 48
  paths = [[(48 - x, y) for x, y in p] if left else p for p in maneuver_paths(kind)]
  return Shape(stroke_tris([[(cx - size / 2 + x * s, cy - size / 2 + y * s) for x, y in p] for p in paths], 4.5 * s))


def draw_maneuver(cx: float, cy: float, size: float, kind: str, left: bool, color: rl.Color) -> None:
  _maneuver(kind, left, q(cx), q(cy), q(size)).draw(color)


@lru_cache(maxsize=64)
def _lane_arrow(kind: str, cx: float, cy: float, size: float) -> Shape:
  s = size / 24

  def P(x, y):
    return (cx - size / 2 + x * s, cy - size / 2 + y * s)
  if kind in ("right", "left"):
    a = [P(8, 21), P(8, 12)] + [P(*p) for p in arc(12, 12, 4, 180, 270, 6)] + [P(19, 8)]
    b = [P(15, 4), P(19, 8), P(15, 12)]
    if kind == "left":
      a, b = [(2 * cx - x, y) for x, y in a], [(2 * cx - x, y) for x, y in b]
    return Shape(stroke_tris([a, b], 2.8 * s))
  return Shape(stroke_tris([[P(12, 20), P(12, 4)], [P(6, 10), P(12, 4), P(18, 10)]], 2.5 * s))


def lane_arrow(cx: float, cy: float, size: float, kind: str, color: rl.Color) -> None:
  """A lane's arrow in a 24-unit box: up, left or right."""
  _lane_arrow(kind, q(cx), q(cy), q(size)).draw(color)


@lru_cache(maxsize=64)
def _split_arrow(right: bool, cx: float, cy: float, size: float) -> tuple[Shape, Shape]:
  s = size / 24

  def P(x, y):
    x = x if right else 24 - x
    return (cx - size / 2 + x * s, cy - size / 2 + y * s)
  straight = Shape(stroke_tris([[P(9, 15), P(9, 4)], [P(4.5, 8.5), P(9, 4), P(13.5, 8.5)]], 2.5 * s))
  turn = Shape(stroke_tris([[P(9, 21), P(9, 15)] + [P(*p) for p in arc(14, 15, 5, 180, 270, 6)] + [P(20, 10)],
                            [P(17, 7), P(20, 10), P(17, 13)]], 2.8 * s))
  return straight, turn


def split_arrow(cx: float, cy: float, size: float, right: bool, straight_col: rl.Color, turn_col: rl.Color) -> None:
  """A lane that goes straight or turns: its straight branch and its turning branch, each in its own colour."""
  straight, turn = _split_arrow(right, q(cx), q(cy), q(size))
  straight.draw(straight_col)
  turn.draw(turn_col)


@lru_cache(maxsize=32)
def _car(cx: float, cy: float, size: float) -> Shape:
  s = size / 24
  return Shape(arrow_tris((cx, cy - 9 * s), (cx - 7 * s, cy + 9 * s), (cx, cy + 5 * s), (cx + 7 * s, cy + 9 * s)))


def draw_car(cx: float, cy: float, size: float, color: rl.Color) -> None:
  """The map's car arrow, pointing up."""
  _car(q(cx), q(cy), q(size)).draw(color)


@lru_cache(maxsize=32)
def _corners(x0: float, y0: float, x1: float, y1: float, rad: float) -> Shape:
  tris = []
  for kx, ky, cx, cy, a0 in [(x0, y0, x0 + rad, y0 + rad, 180), (x1, y0, x1 - rad, y0 + rad, 270),
                             (x1, y1, x1 - rad, y1 - rad, 0), (x0, y1, x0 + rad, y1 - rad, 90)]:
    pts = arc(cx, cy, rad, a0, a0 + 90, 12)
    tris += [[(kx, ky), p, n] for p, n in zip(pts, pts[1:], strict=False)]
  return Shape(tris)


def mask_corners(r: rl.Rectangle, rad: float, color: rl.Color) -> None:
  """Paints the bits of a rectangle outside its rounded corners, for content a rectangular scissor can't round off."""
  rad = min(rad, r.width / 2, r.height / 2)
  _corners(q(r.x), q(r.y), q(r.x + r.width), q(r.y + r.height), q(rad)).draw(color)


@lru_cache(maxsize=16)
def _road_icon(cx: float, cy: float, d: float, step: float, morph: float) -> tuple[Shape, Shape]:
  s = d / 48

  def P(x, y):
    return (cx + (x - 24) * s, cy + (y - 24) * s)
  m = ease(morph)

  def Q(a, b):
    return P(lerp(a[0], b[0], m), lerp(a[1], b[1], m))
  vy, y0, y1 = 2.0, 37.0, 12.0  # vanishing point height, the road's near and far ends (corners inside the disc)

  def G(xb, y):  # a point at height y on the ground line that meets the bottom at xb
    return 24 + (xb - 24) * (y - vy) / (y0 - vy), y
  edges = stroke_tris([[Q(G(8, y0), (15, 33)), Q(G(8, y1), (33, 15))], [Q(G(40, y0), (33, 33)), Q(G(40, y1), (15, 15))]],
                      lerp(2.0, 4.4, m) * s)
  # each dash slides into the next nearer one's place over a step: the nearest leaves at the bottom, a new one
  # arrives at the top; on the ground the far dash is shorter and thinner
  spans = [(46.0, 38.0), (y0 - 1, 27.0), (23.0, 16.5), (12.0, 9.5)]  # off the bottom, near, far, off the top
  dashes = []
  for k in (1, 2, 3):
    (a0, b0), (a1, b1) = spans[k], spans[k - 1]
    ya, yb = min(lerp(a0, a1, step), y0 - 1), max(lerp(b0, b1, step), y1)
    if ya - yb > 0.3:
      dashes.append((ya, yb))
  tris = []
  for xb in (15.5, 32.5):
    for ya, yb in dashes:
      (xa, _), (xz, _) = G(xb, ya), G(xb, yb)
      wa, wz = 1.9 * s * (ya - vy) / (y0 - vy), 1.9 * s * (yb - vy) / (y0 - vy)
      (ax, ay), (bx, by) = P(xa, ya), P(xz, yb)
      tris.append([(ax + wa / 2, ay), (ax - wa / 2, ay), (bx + wz / 2, by), (bx - wz / 2, by)])  # a dash: one quad
  tris += arrow_tris(P(24, 24.5), P(18.5, 35.5), P(24, 32.5), P(29.5, 35.5))  # the car, a little foreshortened
  return Shape(edges), Shape(tris)


def draw_road_icon(cx: float, cy: float, d: float, color: rl.Color, step: float, morph: float, end_red: rl.Color,
                   x_lines: rl.Color) -> None:
  """Navigate on openpilot's icon in a disc of diameter d: a road running away along the ground (all lines meet at a
  vanishing point just above it), solid edges, dashed lane lines and the car's arrow in the middle lane. step 0..1
  moves the dashes one step toward us (a full step looks the same as none). morph 0..1 turns it into the end X: each
  edge swings onto one stroke of the X while the lane lines and the car fade, and the disc turns red."""
  a = rgba(color)[3] / 255
  if morph > 0:
    rl.draw_circle(int(cx), int(cy), d / 2, with_alpha(end_red, a * morph))
  col = mix(color, with_alpha(x_lines, a), morph)
  edges, inner = _road_icon(q(cx), q(cy), q(d), round(step, 2), round(morph, 2))
  edges.draw(col)
  if morph < 0.45:
    inner.draw(with_alpha(col, 1 - morph / 0.45))


@lru_cache(maxsize=8)
def _pin(cx: float, cy: float, pinned: float) -> tuple[Shape, Shape]:
  ang, lift = math.radians(lerp(40, 0, pinned)), lerp(-1, 0, pinned)

  def PN(x, y):
    dx, dy = x - 24, y - 26 + lift
    return cx + (dx * math.cos(ang) - dy * math.sin(ang)) * 2, cy + (2 + dx * math.sin(ang) + dy * math.cos(ang)) * 2
  head = [PN(18.5, 11), PN(29.5, 11), PN(27.5, 20), PN(32, 25), PN(16, 25), PN(20.5, 20)]
  tl, tr, wr, br, bl, wl = head
  fill = Shape([[tl, wl, wr], [tl, wr, tr], [wl, bl, br], [wl, br, wr]])
  outline = Shape(stroke_tris([head + head[:1]], 4.4) + stroke_tris([[PN(24, 25), PN(24, 37)]], 4.8))
  return fill, outline


def draw_pin(cx: float, cy: float, pinned: float, color: rl.Color) -> None:
  """The layout toggle: a pin, leaning and hollow while the map comes and goes (pinned 0), upright and solid when it's
  pinned open (1). Drawn on a 48-unit grid, 2 px a unit, turning about the pin's waist."""
  fill, outline = _pin(q(cx), q(cy), round(pinned, 2))
  if pinned > 0.01:
    fill.draw(with_alpha(color, pinned))
  outline.draw(color)
