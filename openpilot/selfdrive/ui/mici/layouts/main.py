import time

import pyray as rl
import openpilot.cereal.messaging as messaging
from openpilot.selfdrive.ui.mici.layouts.home import MiciHomeLayout
from openpilot.selfdrive.ui.mici.layouts.nav_page import NavMapPage
from openpilot.selfdrive.ui.mici.onroad.nav_card import NavTurnCard
from openpilot.selfdrive.ui.nav.nav_state import NavState
from openpilot.selfdrive.ui.nav.session import NavSession
from openpilot.selfdrive.ui.mici.layouts.settings.settings import SettingsLayout
from openpilot.selfdrive.ui.mici.layouts.offroad_alerts import MiciOffroadAlerts
from openpilot.selfdrive.ui.mici.onroad.augmented_road_view import AugmentedRoadView
from openpilot.selfdrive.ui.ui_state import device, ui_state
from openpilot.selfdrive.ui.mici.layouts.onboarding import OnboardingWindow
from openpilot.selfdrive.ui.body.layouts.onroad import BodyLayout
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.scroller import Scroller
from openpilot.system.ui.lib.application import gui_app


ONROAD_DELAY = 2.5  # seconds
TAP_OPENS_MAP = True  # while a route is active a tap on the driving view opens the map page; False: home, as stock
MAP_PAGE_TIMEOUT = 15  # s untouched on the map page before driving comes back (other pages: the stock 5 s)


def onroad_tap_target(map_page_shown: bool) -> str:
  """Where a tap on the driving view goes: the map page while a route has one, home otherwise (stock)."""
  return "map" if TAP_OPENS_MAP and map_page_shown else "home"


def page_timeout(on_map_page: bool) -> int | None:
  """The interactive timeout to override with: the map page's own, else none (the stock one)."""
  return MAP_PAGE_TIMEOUT if on_map_page else None


def keep_view_shift(page_index: int, current_index: int, appeared: bool, width: float) -> float:
  """The scroll offset change that keeps the page in view still as the page at page_index appears or goes before it.
  A page going while in view leaves the offset alone: the next one slides into its slot."""
  if current_index <= page_index:
    return 0.0
  return -width if appeared else width


class MiciMainLayout(Scroller):
  def __init__(self):
    super().__init__(snap_items=True, spacing=0, pad=0, scroll_indicator=False, edge_shadows=False)

    self._pm = messaging.PubMaster(['bookmarkButton', 'userBookmark'])

    self._prev_onroad = False
    self._prev_standstill = False
    self._onroad_time_delay: float | None = None
    self._setup = False

    # navigation: read once a frame for the driving view's turn card and the map page beside it
    self._nav = NavState()
    self._nav_session = NavSession(self._nav)
    self._nav_page_leaving = False  # the route was ended on the map page: it stays while it scrolls away
    self._nav_page_was_visible = False
    self._on_nav_page = False

    # Initialize widgets
    self._home_layout = MiciHomeLayout()
    self._alerts_layout = MiciOffroadAlerts()
    self._settings_layout = SettingsLayout()
    self._car_onroad_layout = AugmentedRoadView(bookmark_callback=self._on_bookmark_clicked,
                                                nav_card=NavTurnCard(self._nav, self._nav_session))
    self._body_onroad_layout = BodyLayout()
    self._nav_page = NavMapPage(self._nav, self._nav_session, self._on_route_ended)
    self._nav_page.set_visible(self._nav_page_wanted)

    # Initialize widget rects
    for widget in (self._home_layout, self._alerts_layout, self._settings_layout,
                   self._car_onroad_layout, self._body_onroad_layout, self._nav_page):
      # TODO: set parent rect and use it if never passed rect from render (like in Scroller)
      widget.set_rect(rl.Rectangle(0, 0, gui_app.width, gui_app.height))

    self._scroller.add_widgets([
      self._alerts_layout,
      self._home_layout,
      self._nav_page,
      self._car_onroad_layout,
      self._body_onroad_layout,
    ])
    self._scroller.set_reset_scroll_at_show(False)

    # Disable scrolling when onroad is interacting with bookmark, or the map page's road button is held
    self._scroller.set_scrolling_enabled(lambda: not self._car_onroad_layout.is_swiping_left() and
                                         not (self._nav_page.is_visible and self._nav_page.button.held))

    # Set callbacks
    self._setup_callbacks()

    gui_app.add_nav_stack_tick(self._handle_transitions)
    gui_app.add_nav_stack_tick(self._update_nav)
    gui_app.push_widget(self)

    # Start onboarding if terms or training not completed, make sure to push after self
    self._onboarding_window = OnboardingWindow(lambda: gui_app.pop_widgets_to(self))
    if not self._onboarding_window.completed:
      gui_app.push_widget(self._onboarding_window)

    # initialize correct onroad layout
    self._on_body_changed()

  @property
  def _onroad_layout(self) -> Widget:
    # For scroll_to
    return self._body_onroad_layout if ui_state.is_body else self._car_onroad_layout

  def _setup_callbacks(self):
    self._alerts_layout.set_enabled(lambda: self.enabled)
    self._alerts_layout.set_pairing_callback(self._settings_layout.show_pairing)
    self._home_layout.set_callbacks(
      on_settings=lambda: gui_app.push_widget(self._settings_layout),
      on_alerts=lambda: self._scroll_to(self._alerts_layout),
      alert_count_callback=self._alerts_layout.active_alerts,
      alert_icon_callback=self._alerts_layout.highest_severity_icon,
    )
    self._car_onroad_layout.set_click_callback(self._on_onroad_tap)
    self._body_onroad_layout.set_click_callback(lambda: self._scroll_to(self._home_layout))

    device.add_interactive_timeout_callback(self._on_interactive_timeout)
    ui_state.add_on_body_changed_callbacks(self._on_body_changed)

  def _scroll_to(self, layout: Widget):
    layout_x = int(layout.rect.x)
    self._scroller.scroll_to(layout_x, smooth=True)

  def _update_state(self):
    super()._update_state()
    # TODO: Hack to run alert updates while not in view. Add a nav stack tick?
    self._alerts_layout._update_state()
    self._update_nav_page()

  def _update_nav(self):
    # every frame, whatever page or pushed widget shows, so no navRoute update is missed
    self._nav.update(ui_state.sm)
    self._nav_session.update(time.monotonic())

  def _nav_page_wanted(self) -> bool:
    return (ui_state.started and self._nav_session.route_on and not ui_state.is_body) or self._nav_page_leaving

  def _page_in_view(self, page: Widget) -> bool:
    return abs(page.rect.x - self._rect.x) < page.rect.width / 2

  def _update_nav_page(self):
    """The map page comes and goes with the route without moving the page in view, and has a longer timeout."""
    if self._nav_page_leaving and not self._scroller.is_auto_scrolling:
      self._nav_page_leaving = False
    visible = self._nav_page.is_visible
    changed = visible != self._nav_page_was_visible
    if changed:
      self._nav_page_was_visible = visible
      if not visible:
        self._nav_page.button.reset()
      shown_before = [w for w in self._scroller.items if w.is_visible and w is not self._nav_page] + ([] if visible else [self._nav_page])
      current = min(shown_before, key=lambda w: abs(w.rect.x - self._rect.x))
      items = self._scroller.items
      self._scroller.shift_offset(keep_view_shift(items.index(self._nav_page), items.index(current), visible, self._nav_page.rect.width))
    on_page = visible and not changed and self._page_in_view(self._nav_page)  # laid out from the next frame
    if on_page != self._on_nav_page:
      self._on_nav_page = on_page
      device.set_override_interactive_timeout(page_timeout(on_page))

  def _on_onroad_tap(self):
    self._scroll_to(self._nav_page if onroad_tap_target(self._nav_page.is_visible) == "map" else self._home_layout)

  def _on_route_ended(self):
    self._nav_page_leaving = True
    self._scroll_to(self._car_onroad_layout)

  def _render(self, _):
    if not self._setup:
      if self._alerts_layout.active_alerts() > 0:
        self._scroller.scroll_to(self._alerts_layout.rect.x)
      else:
        self._scroller.scroll_to(self._rect.width)
      self._setup = True

    # Render
    super()._render(self._rect)

  def _handle_transitions(self):
    # Don't pop if onboarding
    if gui_app.widget_in_stack(self._onboarding_window):
      return

    if ui_state.started != self._prev_onroad:
      self._prev_onroad = ui_state.started

      # onroad: after delay, pop nav stack and scroll to onroad
      # offroad: immediately scroll to home, but don't pop nav stack (can stay in settings)
      if ui_state.started:
        self._onroad_time_delay = rl.get_time()
      else:
        self._scroll_to(self._home_layout)

    # FIXME: these two pops can interrupt user interacting in the settings
    if self._onroad_time_delay is not None and rl.get_time() - self._onroad_time_delay >= ONROAD_DELAY:
      gui_app.pop_widgets_to(self, lambda: self._scroll_to(self._onroad_layout))
      self._onroad_time_delay = None

    # When car leaves standstill, pop nav stack and scroll to onroad
    CS = ui_state.sm["carState"]
    if not CS.standstill and self._prev_standstill:
      gui_app.pop_widgets_to(self, lambda: self._scroll_to(self._onroad_layout))
    self._prev_standstill = CS.standstill

  def _on_interactive_timeout(self):
    # Don't pop if onboarding
    if gui_app.widget_in_stack(self._onboarding_window):
      return

    if ui_state.started:
      # Don't pop if at standstill
      if not ui_state.sm["carState"].standstill:
        gui_app.pop_widgets_to(self, lambda: self._scroll_to(self._onroad_layout))
    else:
      # Screen turns off on timeout offroad, so pop immediately without animation
      gui_app.pop_widgets_to(self, instant=True)
      self._scroll_to(self._home_layout)

  def _on_bookmark_clicked(self):
    for service in ('bookmarkButton', 'userBookmark'):
      msg = messaging.new_message(service, valid=True)
      self._pm.send(service, msg)

  def _on_body_changed(self):
    self._car_onroad_layout.set_visible(not ui_state.is_body)
    self._body_onroad_layout.set_visible(bool(ui_state.is_body))
