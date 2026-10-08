import pyray as rl

from openpilot.cereal import log
from openpilot.selfdrive.ui.nav.nav_state import NavState
from openpilot.selfdrive.ui.onroad.augmented_road_view import AugmentedRoadView
from openpilot.selfdrive.ui.onroad.nav_panel import NavCard
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.widgets import Widget


class NavSplitView(Widget):
  """The onroad camera view with navigation's card over it, or beside it in the always-on split while a route is
  active (nav_panel.py)."""
  def __init__(self):
    super().__init__()
    self.nav = NavState()
    self.road_view = self._child(AugmentedRoadView())
    self.card = NavCard(self.nav)
    self.road_view.overlay = self.card.draw

  def set_click_callback(self, click_callback):
    # the camera view takes taps on press, unless on the HUD's button; not those on the card or its button either
    def on_click():
      if not self.card.claimed and click_callback is not None:
        click_callback()
    self.road_view.set_click_callback(on_click)

  def _update_state(self):
    self.nav.update(ui_state.sm)

  def _render(self, rect: rl.Rectangle):
    if not ui_state.started:
      self.road_view.render(rect)
      return
    self.card.update(rect)
    self.road_view.set_exp_button_visible(self.card.stock_button_shown)
    alert = self.road_view.alert_renderer.get_alert(ui_state.sm)
    if alert is not None and alert.size == log.SelfdriveState.AlertSize.full:
      # full-screen alerts have the whole screen by design: no card or split while one shows
      self.card.clear_hits()
      self.road_view.overlay = None
      self.road_view.render(rect)
      self.road_view.overlay = self.card.draw
      return
    self.road_view.render(self.card.camera_rect)
