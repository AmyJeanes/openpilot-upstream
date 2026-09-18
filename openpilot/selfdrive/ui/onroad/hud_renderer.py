import json
import math
import os
import time

import pyray as rl
from dataclasses import dataclass
from openpilot.common.constants import CV
from openpilot.selfdrive.ui.onroad.exp_button import ExpButton
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget

# Constants
SET_SPEED_NA = 255
KM_TO_MILE = 0.621371
CRUISE_DISABLED_CHAR = '–'


@dataclass(frozen=True)
class UIConfig:
  header_height: int = 300
  border_size: int = 30
  button_size: int = 192
  set_speed_width_metric: int = 200
  set_speed_width_imperial: int = 172
  set_speed_height: int = 204
  wheel_icon_size: int = 144


@dataclass(frozen=True)
class FontSizes:
  current_speed: int = 176
  speed_unit: int = 66
  max_speed: int = 40
  set_speed: int = 90


@dataclass(frozen=True)
class Colors:
  WHITE = rl.WHITE
  DISENGAGED = rl.Color(145, 155, 149, 255)
  OVERRIDE = rl.Color(145, 155, 149, 255)  # Added
  ENGAGED = rl.Color(128, 216, 166, 255)
  DISENGAGED_BG = rl.Color(0, 0, 0, 153)
  OVERRIDE_BG = rl.Color(145, 155, 149, 204)
  ENGAGED_BG = rl.Color(128, 216, 166, 204)
  GREY = rl.Color(166, 166, 166, 255)
  DARK_GREY = rl.Color(114, 114, 114, 255)
  BLACK_TRANSLUCENT = rl.Color(0, 0, 0, 166)
  WHITE_TRANSLUCENT = rl.Color(255, 255, 255, 200)
  BORDER_TRANSLUCENT = rl.Color(255, 255, 255, 75)
  HEADER_GRADIENT_START = rl.Color(0, 0, 0, 114)
  HEADER_GRADIENT_END = rl.BLANK


UI_CONFIG = UIConfig()
FONT_SIZES = FontSizes()
COLORS = Colors()


class HudRenderer(Widget):
  def __init__(self):
    super().__init__()
    """Initialize the HUD renderer."""
    self.is_cruise_set: bool = False
    self.is_cruise_available: bool = True
    self.set_speed: float = SET_SPEED_NA
    self.speed: float = 0.0
    self.v_ego_cluster_seen: bool = False

    self._font_semi_bold: rl.Font = gui_app.font(FontWeight.SEMI_BOLD)
    self._font_bold: rl.Font = gui_app.font(FontWeight.BOLD)
    self._font_medium: rl.Font = gui_app.font(FontWeight.MEDIUM)

    self._exp_button: ExpButton = ExpButton(UI_CONFIG.button_size, UI_CONFIG.wheel_icon_size)
    self._live: dict | None = None; self._live_t = 0.0  # reproject_c4 fit metrics, written by modeld once a second

  def _update_state(self) -> None:
    """Update HUD state based on car state and controls state."""
    sm = ui_state.sm
    if sm.recv_frame["carState"] < ui_state.started_frame:
      self.is_cruise_set = False
      self.set_speed = SET_SPEED_NA
      self.speed = 0.0
      return

    controls_state = sm['controlsState']
    car_state = sm['carState']

    v_cruise_cluster = car_state.vCruiseCluster
    self.set_speed = (
      controls_state.deprecated.vCruise if v_cruise_cluster == 0.0 else v_cruise_cluster
    )
    self.is_cruise_set = 0 < self.set_speed < SET_SPEED_NA
    self.is_cruise_available = self.set_speed != -1

    if self.is_cruise_set and not ui_state.is_metric:
      self.set_speed *= KM_TO_MILE

    v_ego_cluster = car_state.vEgoCluster
    self.v_ego_cluster_seen = self.v_ego_cluster_seen or v_ego_cluster != 0.0
    v_ego = v_ego_cluster if self.v_ego_cluster_seen else car_state.vEgo
    speed_conversion = CV.MS_TO_KPH if ui_state.is_metric else CV.MS_TO_MPH
    self.speed = max(0.0, v_ego * speed_conversion)

  def _render(self, rect: rl.Rectangle) -> None:
    """Render HUD elements to the screen."""
    # Draw the header background
    rl.draw_rectangle_gradient_v(
      int(rect.x),
      int(rect.y),
      int(rect.width),
      UI_CONFIG.header_height,
      COLORS.HEADER_GRADIENT_START,
      COLORS.HEADER_GRADIENT_END,
    )

    if self.is_cruise_available:
      self._draw_set_speed(rect)

    self._draw_current_speed(rect)

    button_x = rect.x + rect.width - UI_CONFIG.border_size - UI_CONFIG.button_size
    button_y = rect.y + UI_CONFIG.border_size
    self._exp_button.render(rl.Rectangle(button_x, button_y, UI_CONFIG.button_size, UI_CONFIG.button_size))
    self._draw_reproject_debug(rect)

  def _draw_reproject_debug(self, rect: rl.Rectangle) -> None:
    """Debug view of the 3X->comma 4 reprojection's live fits (seam match, rotation, timings) from modeld's live.json."""
    if time.monotonic() - self._live_t > 0.5:
      self._live_t = time.monotonic()
      try:
        self._live = json.load(open('/data/reproject_c4/live.json'))
      except (OSError, ValueError):
        self._live = None
      try:
        self._fit = json.load(open('/data/reproject_c4/fit.json'))
      except (OSError, ValueError):
        self._fit = None
    d = self._live
    if not d:
      return
    f = getattr(self, '_fit', None) or {}
    GREEN, AMBER, RED, BLUE, GREY = rl.Color(128, 216, 166, 255), rl.Color(255, 200, 80, 255), rl.Color(255, 90, 90, 255), rl.Color(110, 180, 255, 255), rl.Color(170, 170, 170, 255)
    def grade(v, ok, warn):
      return GREY if v is None else GREEN if v < ok else AMBER if v < warn else RED
    rot = d.get('rot_deg') or [0, 0, 0]
    fitted = f.get('fitted') if f else d.get('fitted')  # a refit shows in fit.json while modeld still runs the old fit
    pulse = 0.55 + 0.45 * math.sin(time.monotonic() * 2 * math.pi)  # 1 Hz while aligning: it must not be missed
    if fitted:
      rot_tile = ("ROTATION  FITTED", f"{rot[0]:+.2f} / {rot[1]:+.2f}", GREEN, None)
    elif f.get('building') or d.get('loading'):
      rot_tile = ("ALIGNING CAMERAS", "BUILDING", BLUE, 1.0)
    elif f.get('n'):
      rot_tile = ("ALIGNING CAMERAS", f"{f.get('pct', 0):.0f}%", AMBER, f.get('pct', 0) / 100)
    else:
      rot_tile = ("ALIGNING CAMERAS", {'cameras': "CAMERAS", 'model': "LOADING", 'speed': "SPEED", 'straight': "STRAIGHT"}.get(f.get('why'), "WAITING"), AMBER, 0.0)
    seam = d.get('gain'); seam_rel = (seam / d['model_gain']) if seam and d.get('model_gain') else None
    tiles = [  # label, big value, colour
      ("MODEL ms", f"{d.get('model_ms', 0):.0f}", grade(d.get('model_ms'), 46, 50)),
      ("STAGE ms", f"{d.get('stage_ms', 0):.1f}", grade(d.get('stage_ms'), 5.5, 7)),
      ("DROPS %", f"{d.get('drops', 0):.0f}", grade(d.get('drops'), 1, 5)),
      ("SEAM", f"{seam:.2f}" if seam else "-", GREY if seam_rel is None else GREEN if 0.9 < seam_rel < 1.15 else AMBER),
      rot_tile[:3],
    ]
    big, small, pad, gap, th = 72, 26, 16, 12, 6
    h = big + small + 2 * pad + th
    widths = [int(max(measure_text_cached(self._font_bold, value, big).x, measure_text_cached(self._font_medium, label, small).x) + 2 * pad) for label, value, _ in tiles]
    # centred along the top: alerts (the calibration bar) own the bottom, the corners the set-speed box, experimental
    # button and driver-monitoring icon
    x = int(rect.x + (rect.width - (sum(widths) + gap * (len(tiles) - 1))) / 2); y = int(rect.y + UI_CONFIG.border_size)
    for (label, value, col), w in zip(tiles, widths):
      rl.draw_rectangle(x, y, w, h, COLORS.BLACK_TRANSLUCENT)
      if label == rot_tile[0] and rot_tile[3] is not None:  # aligning: pulsing stripe and a progress fill
        rl.draw_rectangle(x, y, int(w * rot_tile[3]), h, rl.Color(col.r, col.g, col.b, 70))
        col = rl.Color(col.r, col.g, col.b, int(255 * pulse))
      rl.draw_rectangle(x, y, w, th, col)
      rl.draw_text_ex(self._font_medium, label, rl.Vector2(x + pad, y + th + pad - 4), small, 0, GREY)
      rl.draw_text_ex(self._font_bold, value, rl.Vector2(x + pad, y + th + pad + small), big, 0, col)
      x += w + gap

  def user_interacting(self) -> bool:
    return self._exp_button.is_pressed

  def _draw_set_speed(self, rect: rl.Rectangle) -> None:
    """Draw the MAX speed indicator box."""
    set_speed_width = UI_CONFIG.set_speed_width_metric if ui_state.is_metric else UI_CONFIG.set_speed_width_imperial
    x = rect.x + 60 + (UI_CONFIG.set_speed_width_imperial - set_speed_width) // 2
    y = rect.y + 45

    set_speed_rect = rl.Rectangle(x, y, set_speed_width, UI_CONFIG.set_speed_height)
    rl.draw_rectangle_rounded(set_speed_rect, 0.35, 10, COLORS.BLACK_TRANSLUCENT)
    rl.draw_rectangle_rounded_lines_ex(set_speed_rect, 0.35, 10, 6, COLORS.BORDER_TRANSLUCENT)

    max_color = COLORS.GREY
    set_speed_color = COLORS.DARK_GREY
    if self.is_cruise_set:
      set_speed_color = COLORS.WHITE
      if ui_state.status == UIStatus.ENGAGED:
        max_color = COLORS.ENGAGED
      elif ui_state.status == UIStatus.DISENGAGED:
        max_color = COLORS.DISENGAGED
      elif ui_state.status == UIStatus.OVERRIDE:
        max_color = COLORS.OVERRIDE

    max_text = tr("MAX")
    max_text_width = measure_text_cached(self._font_semi_bold, max_text, FONT_SIZES.max_speed).x
    rl.draw_text_ex(
      self._font_semi_bold,
      max_text,
      rl.Vector2(x + (set_speed_width - max_text_width) / 2, y + 27),
      FONT_SIZES.max_speed,
      0,
      max_color,
    )

    set_speed_text = CRUISE_DISABLED_CHAR if not self.is_cruise_set else str(round(self.set_speed))
    speed_text_width = measure_text_cached(self._font_bold, set_speed_text, FONT_SIZES.set_speed).x
    rl.draw_text_ex(
      self._font_bold,
      set_speed_text,
      rl.Vector2(x + (set_speed_width - speed_text_width) / 2, y + 77),
      FONT_SIZES.set_speed,
      0,
      set_speed_color,
    )

  def _draw_current_speed(self, rect: rl.Rectangle) -> None:
    """Draw the current vehicle speed and unit."""
    speed_text = str(round(self.speed))
    speed_text_size = measure_text_cached(self._font_bold, speed_text, FONT_SIZES.current_speed)
    speed_pos = rl.Vector2(rect.x + rect.width / 2 - speed_text_size.x / 2, 180 - speed_text_size.y / 2)
    rl.draw_text_ex(self._font_bold, speed_text, speed_pos, FONT_SIZES.current_speed, 0, COLORS.WHITE)

    unit_text = tr("km/h") if ui_state.is_metric else tr("mph")
    unit_text_size = measure_text_cached(self._font_medium, unit_text, FONT_SIZES.speed_unit)
    unit_pos = rl.Vector2(rect.x + rect.width / 2 - unit_text_size.x / 2, 290 - unit_text_size.y / 2)
    rl.draw_text_ex(self._font_medium, unit_text, unit_pos, FONT_SIZES.speed_unit, 0, COLORS.WHITE_TRANSLUCENT)
