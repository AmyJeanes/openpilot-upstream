"""Drawing for the navigation views: thick polylines and the maneuver and lane arrows, as vector strokes so they scale
to any layout."""
import math

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
  """Many polylines of one colour, each of its own width in px, mitred and square-capped. They're laid out once in
  their own coordinates (metres east and north, y up), so a frame only maps them onto the screen (a rotation, a scale
  and the flip of y) and draws a few strips, joined by degenerate triangles."""
  CHUNK = 6000  # strip vertices per draw call, within raylib's batch

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

  def draw(self, origin: np.ndarray, centre: np.ndarray, rot: np.ndarray, scale: float, color: rl.Color) -> None:
    """Draws with `centre` (in the lines' coordinates) at screen point `origin`, rotated by `rot` (2x2, rows giving
    screen right and up) and scaled px per unit."""
    if not self.chunks:
      return
    rel = (self.anchors - centre) @ rot.T * scale + self.offsets @ rot.T * self.halves
    screen = np.empty(rel.shape, np.float32)
    screen[:, 0] = origin[0] + rel[:, 0]
    screen[:, 1] = origin[1] - rel[:, 1]
    for idx in self.chunks:
      s = np.ascontiguousarray(screen[idx])
      rl.draw_triangle_strip(rl.ffi.from_buffer("Vector2[]", s), len(s), color)


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


def arc(cx: float, cy: float, r: float, a0: float, a1: float, n: int = 8) -> list[tuple[float, float]]:
  """Points on a circle (screen angles in degrees, y down) from a0 to a1."""
  return [(cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))) for a in np.linspace(a0, a1, n)]


def bezier(p0, p1, p2, p3, n: int = 10) -> list[tuple[float, float]]:
  t = np.linspace(0.0, 1.0, n)[:, None]
  pts = (1 - t) ** 3 * np.array(p0) + 3 * (1 - t) ** 2 * t * np.array(p1) + 3 * (1 - t) * t ** 2 * np.array(p2) + t ** 3 * np.array(p3)
  return [tuple(p) for p in pts]


Strokes = list[list[tuple[float, float]]]

# Strokes in a 48-unit box, drawn for a right-hand maneuver and mirrored for a left one
_STRAIGHT: Strokes = [[(24, 43), (24, 8)], [(14, 18), (24, 8), (34, 18)]]
_TURN: Strokes = [[(14, 43), (14, 24), *arc(22, 24, 8, 180, 270), (35, 16)], [(28, 9), (35, 16), (28, 23)]]
_SLIGHT: Strokes = [[(18, 44), (18, 26), *bezier((18, 26), (18, 18), (24, 12), (32, 9))], [(25, 6), (33, 8.5), (31, 16.5)]]
_SHARP: Strokes = [[(16, 43), (16, 12), (34, 32)], [(34, 21), (34, 32), (23, 32)]]
_UTURN: Strokes = [[(32, 43), (32, 21), *arc(23, 21, 9, 0, -180), (14, 31)], [(8, 25), (14, 32), (20, 25)]]  # turning left
_MERGE: Strokes = [[(14, 43), (14, 34), (24, 24)], [(34, 43), (34, 34), (24, 24), (24, 8)], [(14, 18), (24, 8), (34, 18)]]
# in at the bottom, round the right side as traffic on the right goes, and out at the top
_ROUNDABOUT: Strokes = [[(18, 45), (18, 37), *arc(18, 28, 9, 90, -90, 14), (18, 5)], [(10, 13), (18, 5), (26, 13)]]


def maneuver_strokes(kind: str, modifier: str) -> tuple[Strokes, bool]:
  """The icon's strokes for a maneuver (OSRM type and modifier), and whether to mirror them for a left one."""
  left = "left" in modifier
  if kind == "arrive":
    return [], False
  if kind == "roundabout" or kind == "exit roundabout":
    return _ROUNDABOUT, False
  if kind == "merge":
    return (_MERGE if modifier == "straight" else _SLIGHT), left
  if modifier == "uturn":
    return _UTURN, False
  if modifier.startswith("sharp"):
    return _SHARP, left
  if modifier.startswith("slight") or kind in ("fork", "off ramp") and modifier != "straight":
    return _SLIGHT, left
  if modifier in ("left", "right"):
    return (_SLIGHT if kind in ("on ramp", "off ramp", "fork") else _TURN), left
  return _STRAIGHT, False


def _place(stroke, rect: rl.Rectangle, box: float, mirror: bool) -> np.ndarray:
  pts = np.array(stroke, np.float32)
  if mirror:
    pts[:, 0] = box - pts[:, 0]
  return pts * np.float32(rect.width / box) + np.float32([rect.x, rect.y])


def draw_maneuver_icon(rect: rl.Rectangle, kind: str, modifier: str, color: rl.Color, thickness: float) -> None:
  if kind == "arrive":
    s = rect.width / 48
    pole = [(14, 44), (14, 6)]
    draw_polyline(_place(pole, rect, 48, False), thickness, color, round_caps=True)
    a, b, c = (rl.Vector2(rect.x + x * s, rect.y + y * s) for x, y in ((14, 6), (14, 24), (38, 15)))
    rl.draw_triangle(a, b, c, color)
    return
  strokes, mirror = maneuver_strokes(kind, modifier)
  for stroke in strokes:
    draw_polyline(_place(stroke, rect, 48, mirror), thickness, color, round_caps=True)


# Lane arrows in a 24-unit box
_LANE: dict[str, tuple[Strokes, bool]] = {
  "straight": ([[(12, 21), (12, 4)], [(6, 10), (12, 4), (18, 10)]], False),
  "right": ([[(8, 21), (8, 12), *arc(12, 12, 4, 180, 270, 5), (19, 8)], [(15, 4), (19, 8), (15, 12)]], False),
  "slightRight": ([[(10, 21), (10, 12), *bezier((10, 12), (10, 8), (13, 5), (17, 4), 6)], [(13, 3), (17.5, 3.5), (17, 8)]], False),
}
_LANE["left"] = (_LANE["right"][0], True)
_LANE["slightLeft"] = (_LANE["slightRight"][0], True)
_ONCOMING: Strokes = [[(12, 4), (12, 20)], [(6, 14), (12, 20), (18, 14)]]


def draw_lane_arrow(rect: rl.Rectangle, direction: str, color: rl.Color, thickness: float, oncoming: bool = False) -> None:
  strokes, mirror = (_ONCOMING, False) if oncoming else _LANE.get(direction, _LANE["straight"])
  for stroke in strokes:
    draw_polyline(_place(stroke, rect, 24, mirror), thickness, color, round_caps=True)


def draw_dashed_rect(rect: rl.Rectangle, radius: float, dash: float, thickness: float, color: rl.Color) -> None:
  """A dashed outline along the straight sides of a rounded rectangle."""
  x0, y0, x1, y1 = rect.x, rect.y, rect.x + rect.width, rect.y + rect.height
  for (ax, ay), (bx, by) in (((x0 + radius, y0), (x1 - radius, y0)), ((x0 + radius, y1), (x1 - radius, y1)),
                             ((x0, y0 + radius), (x0, y1 - radius)), ((x1, y0 + radius), (x1, y1 - radius))):
    length = math.hypot(bx - ax, by - ay)
    n = max(int(length // (2 * dash)), 1)
    step = length / n
    for k in range(n):
      t0, t1 = (k * step + step / 4) / length, (k * step + step * 3 / 4) / length
      rl.draw_line_ex(rl.Vector2(ax + (bx - ax) * t0, ay + (by - ay) * t0), rl.Vector2(ax + (bx - ax) * t1, ay + (by - ay) * t1),
                      thickness, color)
  for cx, cy, a in ((x0 + radius, y0 + radius, 180), (x1 - radius, y0 + radius, 270), (x1 - radius, y1 - radius, 0),
                    (x0 + radius, y1 - radius, 90)):
    rl.draw_ring(rl.Vector2(cx, cy), radius - thickness / 2, radius + thickness / 2, a + 30, a + 60, 4, color)



def draw_corner_masks(rect: rl.Rectangle, radius: float, color: rl.Color, segments: int = 10) -> None:
  """Fills each corner of rect outside its rounding of `radius`, to round off content clipped square to rect."""
  x0, y0, x1, y1 = rect.x, rect.y, rect.x + rect.width, rect.y + rect.height
  # each corner, the centre of its rounding, and where the rounding's quarter circle starts (screen degrees, y down)
  for (cx, cy), (ox, oy), a in (((x0, y0), (x0 + radius, y0 + radius), 180), ((x1, y0), (x1 - radius, y0 + radius), 270),
                                ((x1, y1), (x1 - radius, y1 - radius), 0), ((x0, y1), (x0 + radius, y1 - radius), 90)):
    fan = [(cx, cy), *arc(ox, oy, radius, a + 90, a, segments)]  # this winding faces the camera: raylib culls the other
    rl.draw_triangle_fan([rl.Vector2(px, py) for px, py in fan], len(fan), color)
