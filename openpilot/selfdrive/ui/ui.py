#!/usr/bin/env python3
import os
import time

from openpilot.cereal import messaging
from openpilot.common.hardware import COMMA_HARDWARE, HARDWARE
from openpilot.common.realtime import Priority, config_realtime_process, set_core_affinity
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.layouts.main import MainLayout
from openpilot.selfdrive.ui.mici.layouts.main import MiciMainLayout
from openpilot.selfdrive.ui.ui_state import ui_state

BIG_UI = gui_app.big_ui()

CAMERA_PERIOD_NS = 50_000_000
PRESENT_PHASE_NS = -1_000_000  # just before each frame: after the driving model, before driver monitoring starts
CAMERA_STALE_NS = 200_000_000


class CameraFrameClock:
  """Presents each onroad frame at a fixed point in the road camera's period, where the GPU is free between the driving
  and driver monitoring models. A free-running 20 fps timer drifts through the period and slows whichever model it lands on."""
  def __init__(self):
    self._last = 0.0

  def __call__(self) -> float | None:
    eof = ui_state.sm['narrowRoadCameraState'].timestampEof  # CLOCK_BOOTTIME
    now_ns = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    if not ui_state.started or not eof or now_ns - eof > CAMERA_STALE_NS:
      return None
    deadline = time.monotonic() + (eof + PRESENT_PHASE_NS - now_ns) % CAMERA_PERIOD_NS * 1e-9
    if deadline - self._last < CAMERA_PERIOD_NS * 0.5e-9:
      deadline += CAMERA_PERIOD_NS * 1e-9
    self._last = deadline
    return deadline


def main():
  cores = {5, }
  # above plannerd and radard
  config_realtime_process(0, Priority.CTRL_HIGH)

  gui_app.init_window("UI")
  if BIG_UI:
    MainLayout()
  else:
    MiciMainLayout()

  if HARDWARE.get_device_type() == 'tizi':
    gui_app.set_frame_clock(CameraFrameClock())

  pm = messaging.PubMaster(['uiDebug'])
  for should_render, frame_time, cpu_time in gui_app.render():
    extra_start = time.monotonic()
    ui_state.update()

    if should_render:
      # reaffine after power save offlines our core
      if COMMA_HARDWARE and os.sched_getaffinity(0) != cores:
        try:
          set_core_affinity(list(cores))
        except OSError:
          pass

      extra_cpu = time.monotonic() - extra_start
      msg = messaging.new_message('uiDebug')
      msg.uiDebug.cpuTimeMillis = (cpu_time + extra_cpu) * 1000
      msg.uiDebug.frameTimeMillis = frame_time * 1000
      pm.send('uiDebug', msg)


if __name__ == "__main__":
  main()
