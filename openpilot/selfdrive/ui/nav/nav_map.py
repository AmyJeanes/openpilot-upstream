"""A small heading-up map of the route ahead: the nearby roads, the route on from the car, the car and the next
maneuver, from NavState. The roads are laid out once per navRoute, so a frame maps and draws them in a few calls."""
import math
from dataclasses import dataclass

import numpy as np
import pyray as rl

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.nav.draw import StrokeBatch, draw_polyline
from openpilot.selfdrive.ui.nav.nav_state import NavState


@dataclass(frozen=True)
class MapStyle:
  background = rl.Color(0x0E, 0x10, 0x13, 0xFF)
  road = rl.Color(0x2A, 0x2D, 0x33, 0xFF)
  route = rl.Color(0x3D, 0x7B, 0xFD, 0xFF)
  car = rl.WHITE
  marker = rl.WHITE
  road_px_min: float = 9.0  # a road's stroke: px_min + px_per_m * its width, up to px_max (wider than to scale)
  road_px_per_m: float = 0.7
  road_px_max: float = 26.0
  route_px: float = 15.0
  car_size: float = 34.0  # px, the car arrow's half length
  marker_radius: float = 13.0
  car_at: float = 0.80  # of the view's height from its top
  top_margin: float = 80.0  # px kept above the next maneuver
  view_min: float = 150.0  # m shown ahead of the car, between these, by the next maneuver's distance
  view_max: float = 650.0
  view_extra: float = 1.25


DEFAULT_STYLE = MapStyle()
ZOOM_TC = 1.0  # s, the zoom's filter


class NavMap:
  def __init__(self, style: MapStyle = DEFAULT_STYLE, fps: float = 20.0):
    self.style = style
    self._zoom = FirstOrderFilter(0.0, ZOOM_TC, 1 / fps, initialized=False)  # m ahead in view
    self._roads = StrokeBatch([], [])
    self._roads_version = -1

  def render(self, rect: rl.Rectangle, nav: NavState) -> None:
    st = self.style
    car = nav.car()
    if car is None:
      return
    pos, bearing = car  # already smooth: NavState's PoseTracker
    b = bearing
    m = nav.guidance.maneuver
    ahead = min(max((m.distance if m is not None else st.view_max) * st.view_extra, st.view_min), st.view_max)
    if not self._zoom.initialized:
      self._zoom.x, self._zoom.initialized = ahead, True
    ahead = self._zoom.update(ahead)

    cx, cy = rect.x + rect.width / 2, rect.y + rect.height * st.car_at
    scale = max(cy - rect.y - st.top_margin, 1.0) / ahead  # px per m
    s, c = math.sin(math.radians(b)), math.cos(math.radians(b))
    # local (east, north) m -> screen: right of the heading along x, ahead up
    rot = np.array([[c, -s], [s, c]]) * scale  # rows: right, ahead
    origin = np.array([cx, cy])

    def to_screen(pts: np.ndarray) -> np.ndarray:
      rel = (pts - pos) @ rot.T
      return np.stack([origin[0] + rel[:, 0], origin[1] - rel[:, 1]], axis=1)

    radius = math.hypot(max(cx - rect.x, rect.x + rect.width - cx), max(cy - rect.y, rect.y + rect.height - cy)) / scale + 50.0
    if self._roads_version != nav.version:
      self._roads_version = nav.version
      self._roads = StrokeBatch([r.points for r in nav.roads],
                                [min(st.road_px_min + st.road_px_per_m * r.width, st.road_px_max) for r in nav.roads])
    self._roads.draw(origin, pos, rot / scale, scale, st.road)

    if len(nav.route) >= 2:
      along, seg = nav.along_route(pos, bearing)
      end = along + radius * 1.5
      rest = nav.route[seg + 1:]
      rest = rest[nav.route_along[seg + 1:] < end]
      line = np.vstack([nav.route_point(along)[None, :], rest, nav.route_point(min(end, nav.route_along[-1]))[None, :]])
      draw_polyline(to_screen(line), st.route_px, st.route, round_caps=True)
      if m is not None and m.type != "arrive" and m.distance <= radius * 1.5:
        p = to_screen(nav.route_point(along + m.distance)[None, :])[0]
        rl.draw_circle_v(rl.Vector2(float(p[0]), float(p[1])), st.marker_radius + 3, st.background)
        rl.draw_circle_v(rl.Vector2(float(p[0]), float(p[1])), st.marker_radius, st.marker)
      elif m is not None and m.type == "arrive":
        p = to_screen(nav.route[-1][None, :])[0]
        rl.draw_circle_v(rl.Vector2(float(p[0]), float(p[1])), st.marker_radius + 5, st.background)
        rl.draw_circle_v(rl.Vector2(float(p[0]), float(p[1])), st.marker_radius + 2, st.marker)
        rl.draw_circle_v(rl.Vector2(float(p[0]), float(p[1])), st.marker_radius - 4, st.route)

    self._draw_car(cx, cy)

  def _draw_car(self, cx: float, cy: float) -> None:
    st = self.style
    for size, color in ((st.car_size + 4, st.background), (st.car_size, st.car)):
      tip = rl.Vector2(cx, cy - size)
      notch = rl.Vector2(cx, cy + size * 0.45)
      left, right = rl.Vector2(cx - size * 0.8, cy + size * 0.9), rl.Vector2(cx + size * 0.8, cy + size * 0.9)
      rl.draw_triangle(tip, left, notch, color)
      rl.draw_triangle(tip, notch, right, color)
