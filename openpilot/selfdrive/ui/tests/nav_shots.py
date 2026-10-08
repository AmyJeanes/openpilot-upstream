#!/usr/bin/env python3
"""Screenshots of the onroad nav card in its states, from the real UI fed fake nav messages (nav_fake.py) in this
process, on a virtual clock (20 fps steps, so animations land on the same frames every run) and with scripted touches
going through the card's own gesture handling. Run at the device's size on a test prefix:

  OPENPILOT_PREFIX=uiport BIG=1 SCALE=1 python selfdrive/ui/tests/nav_shots.py --out <dir> [--camera <png>] [scenario ...]

Each scenario starts the UI afresh and saves port_<shot>.png at the times it names. Scenarios named check_* assert
instead (exit status 1 on a failure); check_x_input clicks through the X server (XTest) rather than injecting touches,
so it tests the whole input path at whatever SCALE it runs at (run it at SCALE=0.4, as the desktop UI). The perf scenario saves none: it
times the frames (CPU, real time) through a drive past the turn, the card open with its lanes and map, and the split."""
import argparse
import os
import sys
import time

PREFIX = os.environ.get("OPENPILOT_PREFIX", "")
if not PREFIX or PREFIX == "gta5":
  sys.exit("set OPENPILOT_PREFIX to a test prefix of its own (not the live gta5 one)")
os.makedirs(f"/dev/shm/msgq_{PREFIX}", exist_ok=True)

STEP = 0.05  # s of virtual time a frame
LONG_PRESS = 6  # frames a slide's finger moves over
NAMES = ("none", "start", "open", "slide", "end", "split", "split_start", "turn", "alert", "drag", "long", "arrive", "metric",
         "drive", "speed", "check_x_input", "check_routes", "check_full_alert")
PERF_FROM = 1.0  # s, after the card is up


def scenarios(c):
  """name: [(t, action)], an action a callable of the Context. c: the Context, for its helpers."""
  return {
    "none": [(0.0, c.scene("none")), (1.0, c.shot("no_route"))],
    "start": [(0.0, c.scene("none")), (1.0, c.scene("cruise")), (1.15, c.shot("start_30")), (1.25, c.shot("start_50")),
              (1.35, c.shot("start_70")), (2.0, c.shot("card_small"))],
    "open": [(0.0, c.scene("approach")), (2.0, c.shot("open_lanes_noo_off")), (2.2, c.tap("button")),
             (2.45, c.shot("noo_turning_on")), (3.0, c.shot("open_lanes_noo_on")), (3.2, c.tap("button")),
             (3.45, c.shot("noo_turning_off")), (4.0, c.shot("open_lanes_noo_off_again"))],
    "slide": [(0.0, c.scene("approach")), (2.0, c.slide(237, release=False)), (2.6, c.shot("slide_60")),
              (2.7, c.slide(336, release=True, start=237)), (3.0, c.shot("slide_85_released")), (4.0, c.shot("slid_ended"))],
    "end": [(0.0, c.scene("approach")), (2.0, c.end()), (2.15, c.shot("end_30")), (2.25, c.shot("end_50")),
            (2.35, c.shot("end_70")), (3.0, c.shot("ended"))],
    "split": [(0.0, c.scene("cruise")), (1.0, c.tap("card")), (1.8, c.shot("hand_open_pin")), (2.0, c.tap("pin")),
              (2.25, c.shot("split_50")), (3.0, c.shot("split")), (3.05, c.scene("approach")), (4.0, c.shot("split_lanes")),
              (4.2, c.tap("pin")), (4.45, c.shot("unsplit_50")), (5.0, c.shot("unsplit")), (9.5, c.shot("unsplit_pin_gone"))],
    "split_start": [(0.0, c.scene("none")), (0.5, c.pin()), (1.0, c.scene("cruise")), (1.25, c.shot("split_start_50")),
                    (2.0, c.shot("split_started")), (2.2, c.end()), (2.45, c.shot("split_end_50")), (3.0, c.shot("split_ended"))],
    "turn": [(0.0, c.scene("turn")), (1.5, c.shot("turn_now"))],
    "alert": [(0.0, c.scene("approach")), (0.0, c.alert("Move Right for Exit", "", "small")), (1.5, c.shot("alert_small")),
              (1.6, c.alert("Move Right for Exit", "Signal right to confirm", "mid")), (2.5, c.shot("alert_mid"))],
    "drag": [(0.0, c.scene("cruise")), (1.0, c.drag_card(0.5)), (1.5, c.shot("drag_half")), (2.5, c.shot("drag_snapped"))],
    "long": [(0.0, c.scene("approach", road="long")), (1.5, c.shot("long_name"))],
    "arrive": [(0.0, c.scene("arrive")), (1.5, c.shot("arrive"))],
    "metric": [(0.0, c.metric()), (0.0, c.scene("approach")), (1.5, c.shot("metric"))],
    # driving from 450 m before the turn: the card opens for it by itself, then closes after it
    # real clicks at the window's scale: the nav button toggles NoO, a tap on the camera brings up the sidebar
    "check_x_input": [(0.0, c.scene("approach")), (2.0, c.xtap("button")), (3.0, c.check("NoO on after a click", lambda: c.noo())),
                      (3.2, c.xtap("button")), (4.0, c.check("NoO off after a second click", lambda: not c.noo())),
                      (4.2, c.xtap("camera")), (5.0, c.check("sidebar shown after a click on the camera", lambda: c.sidebar())),
                      (5.2, c.xtap("camera")), (6.0, c.check("sidebar hidden after another", lambda: not c.sidebar())),
                      (6.2, c.xtap("button", hold=0)), (7.0, c.check("NoO on after a quick click", lambda: c.noo()))],
    # a new destination takes NoO from the setting; a reroute (the route sent again, guidance lost a moment) keeps the
    # driver's choice; ending and setting a route again takes the setting again
    "check_routes": [(0.0, c.setting(True)), (0.0, c.scene("approach")), (1.5, c.check("a new route starts with NoO on", lambda: c.noo())),
                     (1.6, c.tap("button")), (2.2, c.check("tapped off", lambda: not c.noo())),
                     (2.3, c.reroute()), (3.5, c.check("a reroute keeps it off", lambda: not c.noo())),
                     (3.6, c.scene("none")), (4.0, c.scene("approach")), (5.5, c.check("guidance lost a moment keeps it off", lambda: not c.noo())),
                     (5.6, c.new_destination()), (8.0, c.check("a new destination starts with NoO on", lambda: c.noo())),
                     (8.1, c.setting(False)), (8.2, c.end()), (9.0, c.new_destination()), (11.0, c.check("off by the setting", lambda: not c.noo()))],
    # a full-screen alert has the whole screen, in the split with the card open
    "check_full_alert": [(0.0, c.pin()), (0.0, c.scene("approach")), (2.0, c.alert("TAKE CONTROL IMMEDIATELY", "Calibration Invalid", "full")),
                         (2.5, c.shot("full_alert_split")), (2.55, c.check("the alert covers the screen", lambda: c.covered()))],
    # navigation's speed cap under the MAX box (35 mph), for each reason; released, it fades; in the split, and metric
    "speed": [(0.0, c.scene("approach")), (0.0, c.speed(5.4, "turnRight")), (1.5, c.shot("speed_turn")),
              (1.6, c.speed(11.2, "bend")), (2.5, c.shot("speed_bend")), (2.6, c.speed(13.0, "speedLimit")), (3.5, c.shot("speed_limit")),
              (3.6, c.speed(9.0, "laneChange")), (4.5, c.shot("speed_lane_change")), (4.6, c.speed(4.5, "bay")),
              (5.5, c.shot("speed_bay")), (5.6, c.scene("arrive")), (5.6, c.speed(3.1, "arrival")), (6.5, c.shot("speed_arrival")),
              (6.6, c.speed(20.0, "bend")), (6.7, c.shot("speed_above_set_fading")), (7.5, c.shot("speed_above_set_hidden")),
              (7.6, c.speed(6.7, "turnLeft")), (7.7, c.shot("speed_fading_in")), (8.5, c.pin()), (9.5, c.shot("speed_split")),
              (9.6, c.metric()), (10.5, c.shot("speed_split_metric")), (10.6, c.speed(0.0, "none")), (11.5, c.shot("speed_released"))],
    "perf": [(0.0, c.drive(450.0)), (30.0, c.pin()), (45.0, c.stop())],
    "drive": [(0.0, c.drive(450.0)), (5.0, c.shot("drive_cruise")), (21.0, c.shot("drive_opening")),
              (25.0, c.shot("drive_lanes")), (36.8, c.shot("drive_turn")), (39.0, c.shot("drive_after_turn")),
              (41.0, c.shot("drive_closed"))],
  }


class Context:
  def __init__(self, out: str, camera: str | None):
    from openpilot.common.params import Params
    from openpilot.common.version import terms_version, training_version
    self.params = Params()
    self.params.put("HasAcceptedTerms", terms_version)
    self.params.put("CompletedTrainingVersion", training_version)
    self.params.put_bool("IsMetric", False)
    self.params.put_bool("ExperimentalModeConfirmed", True)
    self.params.put_bool("NavigateOnOpenpilot", False)
    self.params.put_bool("NavSplitPinned", False)

    from openpilot.selfdrive.ui.tests.nav_fake import FakeCamera, FakeNav, FakeOnroad
    self.out = out
    self.nav, self.op = FakeNav("none"), FakeOnroad()
    self.cam = FakeCamera(camera) if camera else None
    self.frames: list[list] = []  # touch events for the frames to come
    self.card = None
    self.layout = None
    self.failed = False
    self.last_shot = ""
    self.xinput = None
    self.release_in = 0
    self.clock = 0.0
    self.done = False

  # *** actions ***

  def scene(self, name: str, road: str = "short"):
    def f(_):
      self.nav.scene, self.nav.road = name, road
    return f

  def drive(self, before_turn: float):
    def f(_):
      from openpilot.selfdrive.ui.tests.nav_fake import TURN_AT
      self.nav.scene, self.nav.drive_s = "drive", TURN_AT - before_turn
    return f

  def speed(self, cap: float, reason: str):
    def f(_):
      self.nav.speed = (cap, reason)
    return f

  def noo(self) -> bool:
    return self.card.noo and self.params.get_bool("NavigateOnOpenpilot")

  def sidebar(self) -> bool:
    return self.layout._sidebar.is_visible

  def setting(self, on: bool):
    def f(_):
      self.params.put_bool("NavigateOnOpenpilotDefault", on)
    return f

  def reroute(self):
    def f(_):  # a new route to the same destination: its points shift, its end stays
      self.nav.reroute()
    return f

  def new_destination(self):
    def f(_):
      self.nav.new_destination()
      self.nav.scene = "approach"
    return f

  def check(self, what: str, ok):
    def f(rl):
      good = bool(ok())
      self.failed |= not good
      print(f"{self.clock:5.2f} s {'PASS' if good else 'FAIL'}: {what}", flush=True)
    f.is_shot = True
    return f

  def covered(self) -> bool:
    """The camera view, which draws the alert, has the whole screen, and nothing of the card is left to draw or touch."""
    from openpilot.system.ui.lib.application import gui_app
    from openpilot.selfdrive.ui.layouts.main import MainState
    r = self.layout._layouts[MainState.ONROAD].road_view.rect
    c = self.card
    print(f"   camera view {r.x:.0f},{r.y:.0f} {r.width:.0f}x{r.height:.0f}; card hit {c.hit_card.width:.0f}x{c.hit_card.height:.0f}",
          flush=True)
    return (r.x, r.y, r.width, r.height) == (0, 0, gui_app.width, gui_app.height) and c.hit_card.width == 0

  def xtap(self, what: str, hold: int = 3):
    """A click through the X server where `what` is drawn, held `hold` frames (0: released at once, as a quick click
    whose press and release arrive between two frames)."""
    def f(_):
      import pyray as rl
      from openpilot.system.ui.lib.application import gui_app
      x, y = self._point(what)
      wp = rl.get_window_position()
      self.xinput.click(int(wp.x + x * gui_app._scale), int(wp.y + y * gui_app._scale))
      self.release_in = hold
      if not hold:
        self.xinput.release()
    return f

  def stop(self):
    def f(_):
      self.done = True
    return f

  def alert(self, text1: str, text2: str, size: str):
    def f(_):
      self.op.alert = (text1, text2, size)
    return f

  def metric(self):
    def f(_):
      self.params.put_bool("IsMetric", True)
      from openpilot.selfdrive.ui.ui_state import ui_state
      ui_state.is_metric = True
    return f

  def pin(self):
    def f(_):
      self.card.split_on = True
    return f

  def end(self):
    def f(_):  # as the slide to end does once it lets go past the end
      self.card._end_route()
      self.card.route_on = False
    return f

  def shot(self, name: str):
    def f(rl):
      rl.rl_draw_render_batch_active()
      img = rl.load_image_from_screen()
      path = os.path.join(self.out, f"port_{name}.png")
      rl.export_image(img, path)
      self.last_shot = path
      rl.unload_image(img)
      c = self.card
      noo_param = self.params.get_bool("NavigateOnOpenpilot")
      print(f"{self.clock:5.2f} s {name}: route_on {c.route_on} phase {c.phase} open {c.open.b:.0f} noo {c.noo} (param {noo_param})" +
            f" split {c.split_on} slide {c.slide:.0f}/{c.slide_len:.0f}", flush=True)
    f.is_shot = True
    return f

  def _point(self, what: str) -> tuple[float, float]:
    c = self.card
    if what == "button":
      return c.hit_exp[0], c.hit_exp[1]
    if what == "pin":
      return c.hit_layout[0], c.hit_layout[1]
    if what == "camera":
      return c.camera_rect.x + c.camera_rect.width * 0.3, c.camera_rect.y + c.camera_rect.height * 0.6
    r = c.hit_card
    return r.x + r.width / 2, r.y + 80

  def tap(self, what: str):
    def f(_):
      x, y = self._point(what)
      self.frames += [[self._ev(x, y, pressed=True)], [self._ev(x, y, released=True)]]
    return f

  def slide(self, px: float, release: bool, start: float = 0.0):
    """Presses the nav button (or carries on a press already slid `start` px) and slides it left to px."""
    def f(_):
      x0, y = self.card.press_x if start else self._point("button")[0], self._point("button")[1]
      frames = [] if start else [[self._ev(x0, y, pressed=True)]]
      for i in range(1, LONG_PRESS + 1):
        frames.append([self._ev(x0 - start - (px - start) * i / LONG_PRESS, y)])
      if release:
        frames.append([self._ev(x0 - px, y, released=True)])
      self.frames += frames
    return f

  def drag_card(self, frac: float):
    """Drags the small card down by frac of its travel, and lets go."""
    def f(_):
      x, y = self._point("card")
      dy = frac * self.card.drag_range
      frames = [[self._ev(x, y, pressed=True)]]
      for i in range(1, LONG_PRESS + 1):
        frames.append([self._ev(x, y + dy * i / LONG_PRESS)])
      frames += [[self._ev(x, y + dy)]] * 4  # held a moment, for the shot
      frames.append([self._ev(x, y + dy, released=True)])
      self.frames += frames
    return f

  def _ev(self, x: float, y: float, pressed: bool = False, released: bool = False):
    from openpilot.system.ui.lib.application import MouseEvent, MousePos
    return MouseEvent(MousePos(x, y), 0, pressed, released, not released, self.clock)

  # *** running ***

  def publish(self):
    self.nav.send(self.clock, STEP)
    self.op.v = self.nav.position()[1]
    self.op.send()
    if self.cam is not None:
      self.cam.send()


class XInput:
  """Clicks through the X server (XTest), as a mouse on the desktop would."""
  def __init__(self):
    import ctypes
    self.x11 = ctypes.CDLL("libX11.so.6")
    self.xtst = ctypes.CDLL("libXtst.so.6")
    self.x11.XOpenDisplay.restype = ctypes.c_void_p
    self.x11.XFlush.argtypes = [ctypes.c_void_p]
    self.xtst.XTestFakeMotionEvent.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ulong]
    self.xtst.XTestFakeButtonEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]
    self.d = self.x11.XOpenDisplay(None)
    assert self.d, "no X display"

  def click(self, x: int, y: int):
    self.xtst.XTestFakeMotionEvent(self.d, -1, x, y, 0)
    self.xtst.XTestFakeButtonEvent(self.d, 1, 1, 0)
    self.x11.XFlush(self.d)

  def release(self):
    self.xtst.XTestFakeButtonEvent(self.d, 1, 0, 0)
    self.x11.XFlush(self.d)


def timers(view) -> dict[str, list[float]]:
  """Times the card's update, its drawing and its map's, each frame, in real time."""
  perf: dict[str, list[float]] = {"onroad view": [], "card update": [], "card draw": [], "map": []}

  def timed(obj, attr, key):
    fn = getattr(obj, attr)

    def wrapper(*a, **kw):
      t0 = time.perf_counter()
      try:
        return fn(*a, **kw)
      finally:
        perf[key].append(time.perf_counter() - t0)
    setattr(obj, attr, wrapper)
  card = view.card
  timed(view, "render", "onroad view")
  timed(card, "update", "card update")
  timed(card, "draw", "card draw")
  timed(card.map, "render", "map")
  view.road_view.overlay = card.draw
  return perf


def run(name: str, args):
  real_monotonic = time.monotonic
  ctx = Context(args.out, args.camera)
  time.monotonic = lambda: ctx.clock
  try:
    import pyray as rl
    from openpilot.selfdrive.ui.layouts.main import MainLayout, MainState
    from openpilot.selfdrive.ui.ui_state import ui_state
    from openpilot.system.ui.lib.application import gui_app

    script = sorted(scenarios(ctx)[name], key=lambda e: (e[0], getattr(e[1], "is_shot", False)))
    gui_app.init_window(f"nav shots {name}", fps=200)
    layout = MainLayout()
    ctx.card = layout._layouts[MainState.ONROAD].card
    ctx.layout = layout
    if name == "check_x_input":
      ctx.xinput = XInput()
    else:
      gui_app._mouse.get_events = lambda: ctx.frames.pop(0) if ctx.frames else []
    perf = timers(layout._layouts[MainState.ONROAD]) if name == "perf" else None
    while script and script[0][0] <= 0.0 and not getattr(script[0][1], "is_shot", False):
      script.pop(0)[1](ctx)
    ctx.publish()
    ui_state.update()
    for _ in gui_app.render():
      if perf is not None and ctx.clock < PERF_FROM:
        for v in perf.values():
          v.clear()
      if ctx.done:
        break
      while script and script[0][0] <= ctx.clock + 1e-6 and getattr(script[0][1], "is_shot", False):
        script.pop(0)[1](rl)
      if not script:
        break
      ctx.clock = round(ctx.clock + STEP, 6)
      while script and script[0][0] <= ctx.clock + 1e-6 and not getattr(script[0][1], "is_shot", False):
        script.pop(0)[1](ctx)
      ctx.publish()
      ui_state.update()
      if ctx.release_in:
        ctx.release_in -= 1
        if not ctx.release_in:
          ctx.xinput.release()
    gui_app.close()
    if ctx.failed:
      sys.exit(1)
    if perf is not None:
      import numpy as np
      for k, v in perf.items():
        v = np.array(v) * 1000
        print(f"perf {k}: mean {v.mean():.2f} ms, p90 {np.percentile(v, 90):.2f}, max {v.max():.2f} over {len(v)} frames", flush=True)
  finally:
    time.monotonic = real_monotonic


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", required=True)
  ap.add_argument("--camera")
  ap.add_argument("names", nargs="*")
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  names = args.names or list(NAMES)
  if len(names) > 1:  # each afresh, in its own process
    import subprocess
    failed = False
    for n in names:
      print(f"== {n}", flush=True)
      r = subprocess.run([sys.executable, __file__, "--out", args.out] + (["--camera", args.camera] if args.camera else []) + [n], check=False)
      failed |= r.returncode != 0
    sys.exit(1 if failed else 0)
  run(names[0], args)


if __name__ == "__main__":
  main()
