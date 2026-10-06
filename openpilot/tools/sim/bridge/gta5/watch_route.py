#!/usr/bin/env python3
"""Shows the route input the driving model gets (gta5_route_input.py), live from the bridge's shared memory: the NAV
bins, the route's heading ahead as a path, the next maneuver and the next stop line; and below, a preview of route
input v2's lane slots (gta5_lane_slots.py), which the bridge writes to a file of their own."""
import time

import numpy as np
import pyray as rl

from openpilot.selfdrive.modeld.route_input import HEADER, RouteInputReader
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.tools.sim.bridge.gta5 import gta5_lane_slots as slots_mod
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
TARGET = rl.Color(40, 210, 200, 255)
ONCOMING_FILL = rl.Color(70, 64, 66, 255)
HATCH = rl.Color(200, 70, 70, 255)
LANES_H = 112  # px: the lane slots panel


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
  use = [i for i in range(n) if lo - 1e-3 <= (i + 0.5) / n <= hi + 1e-3]
  for i in range(n):
    rl.draw_rectangle_rec(rl.Rectangle(r.x + 8 + i * lane_w, y, lane_w - 3, 22), color if i in use else EMPTY)
  text(f"be in {lanes_text(use, n)}", r.x + 16 + n * lane_w, y + 1, 18)
  text(f"road: {nx.lanes_in} lanes our way, {nx.lanes_out} after (left to right)", r.x + 8, y + 28, 14, DIM)


def lanes_text(use: list[int], n: int) -> str:
  if not use or len(use) == n:
    return "any lane our way" if n > 1 else "the only lane our way"
  if n == 2 or len(use) == 1 and use[0] in (0, n - 1):
    side = "left" if use[0] == 0 else "right"
    return f"the {side} lane" if len(use) == 1 else f"the {len(use)} {side} lanes"
  return f"lane {use[0] + 1} of {n}" if len(use) == 1 else f"lanes {use[0] + 1}-{use[-1] + 1} of {n}"


def draw(vec: np.ndarray, age: float | None, w: int, h: int, lanes: np.ndarray | None = None, lanes_age: float | None = None):
  rl.clear_background(BG)
  d = decode(vec)
  m = 10
  text("Route input", m, 8, 18, TEXT, FontWeight.BOLD)
  status = "no input file" if age is None else f"input {age:.1f} s old" if age < 10 else f"input {age:.0f} s old"
  stale = age is None or age > 1.0
  text(status, w - m - text_w(status, 16), 9, 16, STOP if stale else DIM)
  present = "PRESENT" if d.present else "no route"
  text(present, m + text_w("Route input", 18, FontWeight.BOLD) + 14, 9, 16, GOOD if d.present else DIM)

  draw_lanes(lanes, lanes_age, d.next, rl.Rectangle(m, h - m - LANES_H, w - 2 * m, LANES_H))
  h -= LANES_H + m  # the route input above it
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


def draw_slot(state: int, r: rl.Rectangle):
  """One lane: oncoming hatched, allowed outlined, target filled, each with its direction; a dot for no lane."""
  cx = r.x + r.width / 2
  if state == slots_mod.ONCOMING:
    rl.draw_rectangle_rec(r, ONCOMING_FILL)
    for c in range(6, int(r.width + r.height), 8):  # "/" hatching, clipped to the box by hand
      u0, u1 = max(0.0, c - r.height), min(r.width, float(c))
      rl.draw_line_ex(rl.Vector2(r.x + u0, r.y + c - u0), rl.Vector2(r.x + u1, r.y + c - u1), 2, HATCH)
    y = r.y + r.height / 2
    rl.draw_triangle(rl.Vector2(cx - 5, y - 4), rl.Vector2(cx, y + 4), rl.Vector2(cx + 5, y - 4), TEXT)
  elif state in (slots_mod.ALLOWED, slots_mod.TARGET):
    target = state == slots_mod.TARGET
    if target:
      rl.draw_rectangle_rec(r, TARGET)
    else:
      rl.draw_rectangle_lines_ex(r, 2, TEXT)
    y = r.y + r.height / 2
    rl.draw_triangle(rl.Vector2(cx, y - 5), rl.Vector2(cx - 5, y + 4), rl.Vector2(cx + 5, y + 4), BG if target else DIM)
  else:
    rl.draw_circle_v(rl.Vector2(cx, r.y + r.height / 2), 2, EMPTY)


def draw_lanes(vec: np.ndarray | None, age: float | None, nxt, r: rl.Rectangle):
  """Route input v2's lane slots: the road here and the road out of the next maneuver, as the driver sees them, left to
  right, with the kerb they are counted from marked; from and by where to move into a target lane."""
  rl.draw_rectangle_rec(r, PANEL)
  text("Lane slots (v2 preview, not yet model input)", r.x + 8, r.y + 6, 14, DIM)
  stale = age is None or age > 1.0
  status = "no lane slots file" if age is None else f"{age:.1f} s old" if age < 10 else f"{age:.0f} s old"
  text(status, r.x + r.width - 8 - text_w(status, 14), r.y + 6, 14, STOP if stale else DIM)
  if vec is None or not vec[slots_mod.SIDE]:
    s = "no lane slots file" if vec is None else "stale" if stale else "no route"
    text(s, r.x + 8, r.y + 44, 22, DIM)
    return
  here, out, start, end = slots_mod.decode(vec)
  right = vec[slots_mod.SIDE] > 0
  label_w, bw, gap, bh = 82, 34.0, 4.0, 28.0
  x0 = r.x + 8 + label_w
  strip = slots_mod.SLOTS * (bw + gap) - gap
  rows = ((r.y + 28, "this road", here), (r.y + 28 + bh + 10, "next road", out))
  for y, label, states in rows:
    text(label, r.x + 8, y + 5, 16, TEXT)
    if (states < 0).all():
      text("no lane data", x0 + 8, y + 5, 16, DIM)
      continue
    for k, state in enumerate(states):
      col = slots_mod.SLOTS - 1 - k if right else k  # slot 0 is the kerb lane
      draw_slot(int(state), rl.Rectangle(x0 + col * (bw + gap), y, bw, bh))
  if (here < 0).all() and (out < 0).all():
    return
  kerb_x = x0 + strip + gap / 2 + 1 if right else x0 - gap / 2 - 1
  y0, y1 = rows[0][0] - 3, rows[1][0] + bh + 3
  rl.draw_line_ex(rl.Vector2(kerb_x, y0), rl.Vector2(kerb_x, y1), 3, RIGHT)
  text("kerb", kerb_x - (text_w("kerb", 12) if right else 0), y1 + 1, 12, RIGHT)
  info_x = x0 + strip + 14
  if start is not None:
    # the slots don't know the car's lane, so this says what the route needs, not whether the car complies
    y = rows[0][0]
    if end > 0:
      text(f"move into lit lane from {start:.0f} m" if start > 0 else "move into lit lane now", info_x, y - 2, 14,
           DIM if start > 0 else TARGET)
      text(f"be in by {end:.0f} m", info_x, y + 14, 14, TARGET)
    else:
      text("keep to lit lane", info_x, y + 5, 16, TARGET)
  elif (here == slots_mod.ALLOWED).sum() > 1:
    text("any allowed lane", info_x, rows[0][0] + 5, 16, DIM)
  if nxt is not None and not (out < 0).all():
    text(f"after {'LEFT' if nxt.side == 'L' else 'RIGHT'} turn in {nxt.dist:.0f} m", info_x, rows[1][0] + 5, 16, LEFT if nxt.side == "L" else RIGHT)
  legend_x = r.x + r.width - 8
  for name, state in (("target", slots_mod.TARGET), ("allowed", slots_mod.ALLOWED), ("oncoming", slots_mod.ONCOMING)):
    legend_x -= text_w(name, 12)
    text(name, legend_x, y1 - 1, 12, DIM)
    legend_x -= 22
    draw_slot(state, rl.Rectangle(legend_x, y1 - 3, 18, 16))
    legend_x -= 10


def input_age(reader: RouteInputReader) -> float | None:
  return None if reader.mm is None else max(time.monotonic() - HEADER.unpack_from(reader.mm)[2], 0.0)


if __name__ == "__main__":
  gui_app._width = gui_app._scaled_width = 640
  gui_app._height = gui_app._scaled_height = 360 + LANES_H + 10
  gui_app._scale = 1.0
  gui_app.init_window("Route input", fps=FPS)
  rl.set_window_state(rl.ConfigFlags.FLAG_WINDOW_RESIZABLE)
  rl.set_target_fps(0)  # raylib's frame limiter busy-waits the end of each frame: sleep instead
  reader = RouteInputReader(ROUTE_LEN)
  lanes_reader = RouteInputReader(slots_mod.PREVIEW_LEN, slots_mod.lane_slots_path())
  due = time.monotonic()
  for _ in gui_app.render():
    vec, lanes = reader.read(), lanes_reader.read()
    lanes_age = input_age(lanes_reader)
    draw(vec, input_age(reader), rl.get_screen_width(), rl.get_screen_height(), lanes if lanes_age is not None else None, lanes_age)
    due = max(due + 1 / FPS, time.monotonic())
    time.sleep(max(due - time.monotonic(), 0.0))
