#!/usr/bin/env python3
"""Screenshots of the onroad nav card in its states, from the real UI fed fake nav messages (nav_fake.py) in this
process, on a virtual clock (20 fps steps, so animations land on the same frames every run) and with scripted touches
going through the card's own gesture handling. Run at the device's size on a test prefix:

  OPENPILOT_PREFIX=uiport BIG=1 SCALE=1 python selfdrive/ui/tests/nav_shots.py --out <dir> [--camera <png>] [scenario ...]

Each scenario starts the UI afresh and saves port_<shot>.png at the times it names."""
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
         "drive")


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
    self.clock = 0.0

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
    gui_app._mouse.get_events = lambda: ctx.frames.pop(0) if ctx.frames else []
    while script and script[0][0] <= 0.0 and not getattr(script[0][1], "is_shot", False):
      script.pop(0)[1](ctx)
    ctx.publish()
    ui_state.update()
    for _ in gui_app.render():
      while script and script[0][0] <= ctx.clock + 1e-6 and getattr(script[0][1], "is_shot", False):
        script.pop(0)[1](rl)
      if not script:
        break
      ctx.clock = round(ctx.clock + STEP, 6)
      while script and script[0][0] <= ctx.clock + 1e-6 and not getattr(script[0][1], "is_shot", False):
        script.pop(0)[1](ctx)
      ctx.publish()
      ui_state.update()
    gui_app.close()
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
    for n in names:
      print(f"== {n}", flush=True)
      subprocess.run([sys.executable, __file__, "--out", args.out] + (["--camera", args.camera] if args.camera else []) + [n], check=False)
    return
  run(names[0], args)


if __name__ == "__main__":
  main()
