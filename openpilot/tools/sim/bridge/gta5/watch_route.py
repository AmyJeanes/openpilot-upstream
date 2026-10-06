#!/usr/bin/env python3
"""Shows the route input the driving model gets (gta5_route_input.py), live from the bridge's shared memory: the NAV
bins, the route's heading ahead as a path, the next maneuver and the next stop line."""
import time

import numpy as np
import pyray as rl

from openpilot.selfdrive.modeld.route_input import HEADER, RouteInputReader
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.tools.sim.bridge.gta5.gta5_route_input import BIN_M, BIN_ZERO, HEADING_AHEAD, ROUTE_LEN, decode

FPS = 10
NAV_FROM, NAV_TO = BIN_ZERO - 5, 50  # bins shown: 100 m behind to 500 m ahead
BG = rl.Color(18, 18, 20, 255)
PANEL = rl.Color(30, 30, 34, 255)
EMPTY = rl.Color(48, 48, 54, 255)
DIM = rl.Color(130, 130, 140, 255)
TEXT = rl.Color(235, 235, 240, 255)
LEFT = rl.Color(60, 150, 255, 255)
RIGHT = rl.Color(255, 150, 40, 255)
OTHER = rl.Color(160, 160, 170, 255)
STOP = rl.Color(255, 70, 70, 255)
GOOD = rl.Color(60, 200, 110, 255)
DIR_COLORS = (OTHER, LEFT, RIGHT)


def text(s: str, x: float, y: float, size: int, color=TEXT, weight=FontWeight.MEDIUM):
  rl.draw_text_ex(gui_app.font(weight), s, rl.Vector2(x, y), size, 0, color)


def text_w(s: str, size: int, weight=FontWeight.MEDIUM) -> float:
  return measure_text_cached(gui_app.font(weight), s, size).x


def heading_path(heading: np.ndarray) -> np.ndarray:
  """[11, 2] points (m right, m ahead) from the car through the headings at HEADING_AHEAD, each held over the 10 m before it."""
  step = np.diff(HEADING_AHEAD, prepend=0.0)
  rad = np.radians(heading)
  return np.vstack([[0.0, 0.0], np.cumsum(np.stack([step * np.sin(rad), step * np.cos(rad)], axis=-1), axis=0)])


def draw_heading(d, r: rl.Rectangle):
  rl.draw_rectangle_rec(r, PANEL)
  text("HEADING", r.x + 8, r.y + 6, 14, DIM)
  car = rl.Vector2(r.x + r.width / 2, r.y + r.height - 18)
  pts = heading_path(d.heading)
  # shrink to fit a sharp bend
  half, up = r.width / 2 - 12, car.y - r.y - 26
  scale = min(up / 100.0, half / max(np.abs(pts[:, 0]).max(), 1e-3), up / max(pts[:, 1].max(), 1e-3))
  rl.begin_scissor_mode(int(r.x), int(r.y), int(r.width), int(r.height))
  for m in (25, 50, 100):
    rad = m * scale
    if rad < up + 10:
      rl.draw_circle_lines(int(car.x), int(car.y), rad, EMPTY)
      text(f"{m} m", car.x + 3, car.y - rad + 1, 12, EMPTY)
  screen = [rl.Vector2(car.x + x * scale, car.y - y * scale) for x, y in pts]
  for a, b in zip(screen, screen[1:], strict=False):
    rl.draw_line_ex(a, b, 4, GOOD)
  for p in screen[1:]:
    rl.draw_circle_v(p, 3, TEXT)
  rl.end_scissor_mode()
  rl.draw_triangle(rl.Vector2(car.x, car.y - 10), rl.Vector2(car.x - 7, car.y + 8), rl.Vector2(car.x + 7, car.y + 8), TEXT)
  far = f"{d.heading[-1]:+.0f}° at {HEADING_AHEAD[-1]:.0f} m"
  text(far, r.x + r.width - text_w(far, 16) - 8, r.y + 6, 16)


def draw_nav(d, r: rl.Rectangle):
  text("NAV", r.x, r.y, 14, DIM)
  legend_x = r.x + r.width
  for name, color in (("other", OTHER), ("right", RIGHT), ("left", LEFT)):
    legend_x -= text_w(name, 13) + 4
    text(name, legend_x, r.y + 1, 13, DIM)
    legend_x -= 14
    rl.draw_rectangle(int(legend_x), int(r.y + 3), 10, 10, color)
    legend_x -= 10
  bar = rl.Rectangle(r.x, r.y + 20, r.width, 28)
  span = (NAV_TO - NAV_FROM) * BIN_M
  bin_w = bar.width / (NAV_TO - NAV_FROM)

  def x_at(m: float) -> float:
    return bar.x + (m + (BIN_ZERO - NAV_FROM) * BIN_M) / span * bar.width

  for i, b in enumerate(range(NAV_FROM, NAV_TO)):
    x = bar.x + i * bin_w
    lit = np.flatnonzero(d.nav[b])
    if not len(lit):
      rl.draw_rectangle_rec(rl.Rectangle(x + 1, bar.y, bin_w - 2, bar.height), EMPTY)
    for j, k in enumerate(lit):  # several in one bin share it, stacked
      h = bar.height / len(lit)
      rl.draw_rectangle_rec(rl.Rectangle(x + 1, bar.y + j * h, bin_w - 2, h), DIR_COLORS[k])
  if d.stop is not None and -BIN_M * (BIN_ZERO - NAV_FROM) <= d.stop <= span:
    sx = x_at(d.stop)
    rl.draw_line_ex(rl.Vector2(sx, bar.y - 4), rl.Vector2(sx, bar.y + bar.height + 4), 3, STOP)
  cx = x_at(0.0)
  rl.draw_line_ex(rl.Vector2(cx, bar.y - 6), rl.Vector2(cx, bar.y + bar.height + 6), 2, TEXT)
  rl.draw_triangle(rl.Vector2(cx - 6, bar.y - 12), rl.Vector2(cx, bar.y - 4), rl.Vector2(cx + 6, bar.y - 12), TEXT)
  for m in range(-100, 501, 100):
    x = x_at(m)
    rl.draw_line_ex(rl.Vector2(x, bar.y + bar.height), rl.Vector2(x, bar.y + bar.height + 5), 1, DIM)
    s = f"{m}"
    text(s, min(max(x - text_w(s, 13) / 2, bar.x), bar.x + bar.width - text_w(s, 13)), bar.y + bar.height + 6, 13, DIM)


def draw_next(d, r: rl.Rectangle):
  rl.draw_rectangle_rec(r, PANEL)
  text("NEXT", r.x + 8, r.y + 6, 14, DIM)
  nx = d.next
  if nx is None:
    text("none within 500 m", r.x + 8, r.y + 28, 24, DIM)
    return
  color = LEFT if nx.side == "L" else RIGHT
  head = f"{'LEFT' if nx.side == 'L' else 'RIGHT'} {abs(nx.change):.0f}°"
  text(head, r.x + 8, r.y + 24, 32, color, FontWeight.BOLD)
  text(f"in {nx.dist:.0f} m", r.x + 16 + text_w(head, 32, FontWeight.BOLD), r.y + 32, 24)
  text(f"junction entry {nx.entry:.0f} m", r.x + 8, r.y + 66, 18)
  y = r.y + 94
  if not nx.lane_data:
    text(f"lanes {nx.lanes_in} -> {nx.lanes_out}, no lane data", r.x + 8, y, 18, DIM)
    return
  n = max(nx.lanes_in, 1)
  lo, hi = nx.target
  lane_w = min(28.0, (r.width - 16) / n / 2)
  for i in range(n):
    on = lo - 1e-3 <= (i + 0.5) / n <= hi + 1e-3
    rl.draw_rectangle_rec(rl.Rectangle(r.x + 8 + i * lane_w, y, lane_w - 3, 22), color if on else EMPTY)
  text(f"lanes {nx.lanes_in} -> {nx.lanes_out}, target {lo:.2f}–{hi:.2f}", r.x + 16 + n * lane_w, y + 1, 18)


def draw(vec: np.ndarray, age: float | None, w: int, h: int):
  rl.clear_background(BG)
  d = decode(vec)
  m = 10
  text("Route input", m, 8, 18, TEXT, FontWeight.BOLD)
  status = "no input file" if age is None else f"input {age:.1f} s old" if age < 10 else f"input {age:.0f} s old"
  stale = age is None or age > 1.0
  text(status, w - m - text_w(status, 16), 9, 16, STOP if stale else DIM)
  present = "PRESENT" if d.present else "no route"
  text(present, m + text_w("Route input", 18, FontWeight.BOLD) + 14, 9, 16, GOOD if d.present else DIM)

  top = 36
  side = min(w * 0.4, h - top - m)
  x0 = m + side + m
  rw = w - x0 - m
  heading_r, next_r = rl.Rectangle(m, top, side, h - top - m), rl.Rectangle(x0, top + 84, rw, h - top - 84 - 52)
  draw_nav(d, rl.Rectangle(x0, top, rw, 70))
  if vec.any():
    draw_heading(d, heading_r)
    draw_next(d, next_r)
    stop = f"STOP line {d.stop:.0f} m" if d.stop is not None else "stop line -"
    text(stop, x0, h - m - 30, 26, STOP if d.stop is not None else DIM, FontWeight.BOLD)
  else:
    rl.draw_rectangle_rec(heading_r, PANEL)
    rl.draw_rectangle_rec(next_r, PANEL)
    s = "NO ROUTE"
    text(s, (w - text_w(s, 64, FontWeight.BOLD)) / 2, h / 2 - 34, 64, TEXT, FontWeight.BOLD)
    why = "no input file" if age is None else "input stale (bridge not writing)" if stale else "bridge sends zeros (no route or off it)"
    text(why, (w - text_w(why, 18)) / 2, h / 2 + 40, 18, DIM)


def input_age(reader: RouteInputReader) -> float | None:
  return None if reader.mm is None else max(time.monotonic() - HEADER.unpack_from(reader.mm)[2], 0.0)


if __name__ == "__main__":
  gui_app._width = gui_app._scaled_width = 640
  gui_app._height = gui_app._scaled_height = 360
  gui_app._scale = 1.0
  gui_app.init_window("Route input", fps=FPS)
  rl.set_window_state(rl.ConfigFlags.FLAG_WINDOW_RESIZABLE)
  rl.set_target_fps(0)  # raylib's frame limiter busy-waits the end of each frame: sleep instead
  reader = RouteInputReader(ROUTE_LEN)
  due = time.monotonic()
  for _ in gui_app.render():
    vec = reader.read()
    draw(vec, input_age(reader), rl.get_screen_width(), rl.get_screen_height())
    due = max(due + 1 / FPS, time.monotonic())
    time.sleep(max(due - time.monotonic(), 0.0))
