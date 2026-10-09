"""The car's turn signals on the onroad HUD, as sunnypilot's: a green arrow either side of the speed's unit flashes while that
blinker is on, and an orange car icon takes its place, steady, while that side's blind spot is occupied."""
import pyray as rl

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app

BLINK_HZ = 1.5
OFFSET_X = 80  # from the HUD's centre to each icon's box
TOP = 190
SIZE = 150


class TurnSignal:
  def __init__(self, right: bool):
    self.blindspot = False
    self.active = False
    self._next_flash = 0
    self._alpha = FirstOrderFilter(0.0, 0.3, 1 / gui_app.target_fps)
    self._arrow = gui_app.texture("icons_mici/onroad/turn_signal_left.png", 120, 109, flip_x=right)
    self._car = gui_app.texture("icons_mici/onroad/blind_spot_left.png", 120, 109, flip_x=right)

  def update(self, blinker: bool, blindspot: bool) -> None:
    blindspot = blindspot and ui_state.show_blindspot
    blinker = blinker and ui_state.show_turn_signals
    if not self.active or self.blindspot != blindspot:
      self._next_flash = gui_app.frame  # the first flash shows at once
    self.active, self.blindspot = blinker or blindspot, blindspot

  def draw(self, rect: rl.Rectangle) -> None:
    if not self.active:
      return
    if self.blindspot:
      texture, alpha = self._car, 255
    else:
      # timed by frames: the filter steps once a frame too
      if gui_app.frame >= self._next_flash:
        self._next_flash = gui_app.frame + max(round(gui_app.target_fps / BLINK_HZ), 1)
        self._alpha.x = 255 * 2  # past full, so each flash holds a moment before fading to the dim level
      else:
        self._alpha.update(255 * 0.2)
      texture, alpha = self._arrow, int(min(self._alpha.x, 255))
    pos = rl.Vector2(rect.x + (rect.width - texture.width) / 2, rect.y + (rect.height - texture.height) / 2)
    rl.draw_texture_ex(texture, pos, 0.0, 1.0, rl.Color(255, 255, 255, alpha))


class TurnSignals:
  """Drawn by the HUD: update() each frame, then draw() over the camera."""
  def __init__(self):
    self._left = TurnSignal(right=False)
    self._right = TurnSignal(right=True)

  def update(self, started: bool) -> None:
    cs = ui_state.sm["carState"]
    self._left.update(started and cs.leftBlinker, started and cs.leftBlindspot)
    self._right.update(started and cs.rightBlinker, started and cs.rightBlindspot)

  def draw(self, rect: rl.Rectangle) -> None:
    x = rect.x + rect.width / 2
    self._left.draw(rl.Rectangle(x - OFFSET_X - SIZE, rect.y + TOP, SIZE, SIZE))
    self._right.draw(rl.Rectangle(x + OFFSET_X, rect.y + TOP, SIZE, SIZE))
