import pyray as rl

from openpilot.selfdrive.ui import UI_BORDER_SIZE
from openpilot.selfdrive.ui.nav.nav_state import NavState
from openpilot.selfdrive.ui.onroad.augmented_road_view import AugmentedRoadView
from openpilot.selfdrive.ui.onroad.nav_panel import NavPanel
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import Widget

CAMERA_SHARE = 0.6  # of the width the camera view keeps beside the navigation panel


class NavSplitView(Widget):
  """The onroad camera view, with the navigation panel to its right while a route is active and the view has the
  whole screen (not beside the sidebar)."""
  def __init__(self):
    super().__init__()
    self.nav = NavState()
    self.road_view = self._child(AugmentedRoadView())
    self.panel = self._child(NavPanel(self.nav))

  def set_click_callback(self, click_callback):
    # the views take the taps themselves: the camera view on press, unless it's on the HUD's buttons
    self.road_view.set_click_callback(click_callback)
    self.panel.set_click_callback(click_callback)

  def _update_state(self):
    self.nav.update(ui_state.sm)

  def _render(self, rect: rl.Rectangle):
    if not (ui_state.started and self.nav.active and rect.width >= gui_app.width * 0.9):
      self.road_view.render(rect)
      return
    split = round(rect.width * CAMERA_SHARE)
    self.road_view.render(rl.Rectangle(rect.x, rect.y, split, rect.height))
    self.panel.render(rl.Rectangle(rect.x + split, rect.y + UI_BORDER_SIZE, rect.width - split - UI_BORDER_SIZE,
                                   rect.height - 2 * UI_BORDER_SIZE))
