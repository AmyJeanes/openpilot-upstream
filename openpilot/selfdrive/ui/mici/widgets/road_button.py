"""The map page's road button: Navigate on openpilot's toggle and the way to end the route. Tap it to turn Navigate on
openpilot on or off (the 3X's road icon: lit blue on, grey off, its lane dashes moving one step toward you as it turns
on). Slide it left to end navigation, mici's slider grammar: it idles small, grows to slider size as the slide starts
while its "slide to end navigation" track and label appear behind it, and its road turns into the red X by the end
point. Let go past that point and it finishes its travel and ends the route; let go before it and it springs back."""
import math
import time

import pyray as rl

from openpilot.selfdrive.ui.nav.draw import Anim, clamp01, draw_road_icon, lerp, mix, with_alpha
from openpilot.selfdrive.ui.nav.session import NavSession
from openpilot.system.ui.lib.application import FONT_SCALE, FontWeight, MouseEvent, gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget

# Sizes in px (300 ppi, 0.085 mm a px). Touch: mici's smallest stock target is the small slider's knob, 115 px tall,
# and a fingertip covers about 110 px, so the button takes a 120 px circle (10 mm) however small it's drawn. Drawn: 60
# px at rest (5 mm; its road symbol 23 arcmin at 700 mm, above the driving view's 14-20 arcmin lane arrows and icon),
# out of the way of the map; 96 px while sliding, the slider's knob and track height, as the approved mockup.
TOUCH_D = 120.0
IDLE_D = 60.0
SLIDE_D = 96.0
PRESSED_SCALE = 1.07  # held, as mici's sliders
GROW_PX = 40.0  # px of slide over which it grows from IDLE_D to SLIDE_D
TAP_SLOP = 20.0  # px the finger may move and still tap
SLIDE_END = 0.75  # let go past this much of the track to end the route; the X is complete here
SHOW_TRACK_PX = 30.0  # px of slide over which the track and its label fade in
FINISH_S = 0.35  # let go past the end point: the button finishes its travel, then the route ends
SPRING_TC = 0.06  # s, the spring back's time constant
LIT_S = 0.5  # Navigate on openpilot's on/off animation

DISC = rl.Color(0x1b, 0x1c, 0x20, 245)
SLIDE_DISC = rl.Color(0x2a, 0x2b, 0x30, 255)
TRACK = rl.Color(0x0c, 0x0c, 0x0e, 255)
OFF_GREY = rl.Color(0x8c, 0x8c, 0x96, 255)
LIT_BLUE = rl.Color(0x5b, 0x8d, 0xff, 255)
END_RED = rl.Color(0xC9, 0x22, 0x31, 255)


class SlideToEnd:
  """The button's press: a tap, or a slide left along a track `length` px long (to the button's travel end)."""
  def __init__(self):
    self.held = False
    self.dragged = False
    self.offset = 0.0  # px slid left
    self.length = 1.0
    self._x0 = 0.0
    self.finish_t: float | None = None  # let go past the end point: finishing the travel since

  def press(self, x: float) -> None:
    if self.finish_t is None:
      self.held, self.dragged, self._x0 = True, False, x

  def move(self, x: float) -> None:
    if self.held:  # 1:1 with the finger, as mici's sliders; moved past the slop it's a slide, not a tap
      self.dragged = self.dragged or abs(x - self._x0) > TAP_SLOP
      self.offset = min(self.length, max(0.0, self._x0 - x))

  def release(self, now: float) -> str:
    """tap, end (past the end point: the route ends once the travel finishes), back (springs back) or '' (no press)."""
    if not self.held:
      return ""
    self.held = False
    if not self.dragged:
      return "tap"
    if self.offset >= SLIDE_END * self.length:
      self.finish_t = now
      return "end"
    return "back"

  def step(self, now: float, dt: float) -> bool:
    """Moves the button on by itself; True once, when a confirmed slide has finished its travel."""
    if self.finish_t is not None:
      self.offset += (self.length - self.offset) * (1 - math.exp(-dt / SPRING_TC))
      if now - self.finish_t >= FINISH_S:
        self.finish_t, self.offset = None, 0.0
        return True
    elif not self.held and self.offset > 0:
      self.offset *= math.exp(-dt / SPRING_TC)
      if self.offset < 1.0:
        self.offset = 0.0
    return False

  @property
  def morph(self) -> float:
    """0 the road, 1 the red X: complete at the end point."""
    return clamp01(self.offset / (SLIDE_END * self.length))

  @property
  def grow(self) -> float:
    """0 drawn at rest size, 1 at slide size."""
    return 1.0 if self.finish_t is not None else clamp01(self.offset / GROW_PX)


class RoadButton(Widget):
  """Its rect is its touch area, centred on the button; the track runs left from it to track_left."""
  def __init__(self, session: NavSession, on_end):
    super().__init__()
    self.session = session
    self._on_end = on_end
    self.slide = SlideToEnd()
    self.track_left = 0.0
    self._semi = gui_app.font(FontWeight.SEMI_BOLD)
    self._lit = Anim(0.0)
    self._pressed = Anim(0.0)
    self._t: float | None = None
    self._label_cache: tuple = ()

  @property
  def held(self) -> bool:
    return self.slide.held

  def reset(self) -> None:
    """Off screen mid-press (the route went): the release won't come here."""
    self.slide = SlideToEnd()

  @property
  def centre(self) -> tuple[float, float]:
    return self._rect.x + self._rect.width / 2, self._rect.y + self._rect.height / 2

  def _handle_mouse_event(self, ev: MouseEvent) -> None:
    cx, cy = self.centre
    if ev.left_pressed:
      if (ev.pos.x - cx) ** 2 + (ev.pos.y - cy) ** 2 <= (TOUCH_D / 2) ** 2:
        self.slide.press(ev.pos.x)
    elif ev.left_released:
      result = self.slide.release(time.monotonic())
      if result == "tap" and self.session.route_on:
        self.session.set_noo(not self.session.noo)
    elif ev.left_down:
      self.slide.move(ev.pos.x)

  def _update_state(self) -> None:
    now = time.monotonic()
    dt = min(max(now - self._t, 0.0), 0.1) if self._t is not None else 0.0
    self._t = now
    cx, _ = self.centre
    self.slide.length = max(1.0, (cx - SLIDE_D / 2) - self.track_left)
    if self.slide.step(now, dt) and self.session.route_on:
      self.session.end()
      self._on_end()
    self._lit.set(1.0 if self.session.noo else 0.0, now, LIT_S)
    self._pressed.set(1.0 if self.slide.held else 0.0, now, 0.1)

  def _label_layout(self, width: float) -> tuple[int, rl.Vector2]:
    """The label's size: 34 px, shrunk to fit before the button down to 26, then left to run under it."""
    key = (width // 8,)
    if self._label_cache[:1] != key:
      label = tr("slide to end navigation")
      em = 34.0
      while em > 26 and measure_text_cached(self._semi, label, round(em / FONT_SCALE)).x > width:
        em -= 1
      size = round(em / FONT_SCALE)
      self._label_cache = (key[0], label, size, measure_text_cached(self._semi, label, size))
    return self._label_cache[2], self._label_cache[3]

  def _render(self, _) -> None:
    now = time.monotonic()
    s = self.slide
    cx, cy = self.centre
    d = lerp(IDLE_D, SLIDE_D, s.grow) * lerp(1.0, PRESSED_SCALE, self._pressed.value(now))
    shown = clamp01(s.offset / SHOW_TRACK_PX)
    if shown > 0:
      track = rl.Rectangle(self.track_left, cy - SLIDE_D / 2, cx + SLIDE_D / 2 - self.track_left, SLIDE_D)
      rl.draw_rectangle_rounded(track, 1.0, 24, with_alpha(TRACK, shown))
      size, lz = self._label_layout(track.width - SLIDE_D * 1.4)
      frac = clamp01(s.offset / s.length)
      rl.draw_text_ex(self._semi, self._label_cache[1], rl.Vector2(track.x + SLIDE_D * 0.35, cy - lz.y / 2), size, 0,
                      with_alpha(rl.WHITE, 0.8 * shown * (1 - frac)))
    bx = cx - s.offset
    rl.draw_circle(int(bx), int(cy), d / 2, mix(DISC, SLIDE_DISC, shown))
    lit = self._lit.value(now)
    step = lit if self._lit.b > 0.5 else 0.0  # turning on, the dashes move one step toward you; turning off, still
    draw_road_icon(bx, cy, d, mix(OFF_GREY, LIT_BLUE, lit), step, s.morph, END_RED, rl.WHITE)
