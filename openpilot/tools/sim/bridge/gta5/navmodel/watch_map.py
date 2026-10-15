#!/usr/bin/env python3
"""Shows the nav model's two inputs live, as modeld gets them: mapsd's map frame (VisionIPC VISION_STREAM_MAP, 256x256,
drawn 2x) and navInstruction's maneuvers, the first of which becomes the `navigation` vector (its Valhalla type and
distance / 100). The frame's age is in the corner; a black frame means mapsd has no route or fix yet."""
import time

import numpy as np
import pyray as rl

import openpilot.cereal.messaging as messaging
from msgq.visionipc import VisionIpcClient
from openpilot.cereal.visionipc import VisionStreamType
from openpilot.system.ui.lib.application import FontWeight, gui_app

FPS = 20
SCALE = 2
SIZE = 256
SIDE_W = 300  # px: the maneuver list
BG = rl.Color(18, 18, 20, 255)
TEXT = rl.Color(235, 235, 240, 255)
DIM = rl.Color(130, 130, 140, 255)
STALE = rl.Color(255, 90, 70, 255)
STALE_S = 0.5  # s since mapsd's last frame


def text(s: str, x: float, y: float, size: int, color=TEXT):
  rl.draw_text_ex(gui_app.font(FontWeight.MEDIUM), s, rl.Vector2(x, y), size, 0, color)


def main():
  gui_app._width = gui_app._scaled_width = SIZE * SCALE + SIDE_W
  gui_app._height = gui_app._scaled_height = SIZE * SCALE
  gui_app._scale = 1.0
  gui_app.init_window("Nav model inputs", fps=FPS)
  rl.set_target_fps(0)  # raylib's frame limiter busy-waits the end of each frame: sleep instead
  tex = rl.load_texture_from_image(rl.Image(None, SIZE, SIZE, 1, rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAYSCALE))
  client = VisionIpcClient("mapsd", VisionStreamType.VISION_STREAM_MAP, True)
  sm = messaging.SubMaster(["navInstruction"])
  last, frame_id, due = 0.0, None, time.monotonic()
  for _ in gui_app.render():
    if not client.is_connected():
      client.connect(False)
    elif (buf := client.recv(0)) is not None:
      img = np.ascontiguousarray(np.frombuffer(buf.data, dtype=np.uint8)[:SIZE * SIZE])
      rl.update_texture(tex, rl.ffi.cast("void *", rl.ffi.from_buffer(img)))
      last, frame_id = time.monotonic(), client.frame_id
    sm.update(0)
    rl.clear_background(BG)
    rl.draw_texture_ex(tex, rl.Vector2(0, 0), 0.0, SCALE, rl.WHITE)
    age = time.monotonic() - last if last else None
    text("no map frames" if age is None else f"frame {frame_id}  {age * 1000:.0f} ms", 8, 6, 16,
         STALE if age is None or age > STALE_S else DIM)
    x, y = SIZE * SCALE + 14, 12
    text("navigation", x, y, 22)
    y += 34
    ni = sm["navInstruction"]
    if sm.recv_frame["navInstruction"] == 0:
      text("no navInstruction", x, y, 18, DIM)
    else:
      text(f"{ni.maneuverPrimaryText or '-'}", x, y, 18)
      y += 26
      text(f"in {ni.maneuverDistance:.0f} m", x, y, 18, DIM)
      y += 36
      for i, m in enumerate(list(ni.allManeuvers)[:12]):
        text(f"{m.distance:6.0f} m  {m.type} {m.modifier}", x, y, 16, TEXT if i == 0 else DIM)
        y += 22
    due = max(due + 1 / FPS, time.monotonic())
    time.sleep(max(due - time.monotonic(), 0.0))


if __name__ == "__main__":
  main()
