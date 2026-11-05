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
MODEL_WAIT_TIMEOUT_MS = 100


class ModelFrameClock:
  """Starts each onroad frame when the model output for the newest camera frame arrives.

  The UI draws the camera frame the model output came from under it, and this keeps that frame the newest one. While the
  model isn't running, frames keep the fixed rate.
  """
  def __init__(self):
    self._sock = messaging.sub_sock('modelV2', conflate=True, timeout=MODEL_WAIT_TIMEOUT_MS)
    self._running = False
    self._frame_id = 0

  def wait(self) -> bool:
    if not ui_state.started:
      self._running = False
      self._frame_id = 0
      return False
    model = messaging.recv_one_or_none(self._sock)
    # output for an earlier frame can still be queued: skip it
    if (model is None or model.modelV2.frameId <= self._frame_id) and self._running:
      model = messaging.recv_one(self._sock)
    self._running = model is not None
    if model is not None:
      self._frame_id = model.modelV2.frameId
    return self._running


def main():
  cores = {5, }
  # above plannerd and radard
  config_realtime_process(0, Priority.CTRL_HIGH)

  gui_app.init_window("UI")
  # update before drawing, so the overlay is the newest model output
  gui_app.add_nav_stack_tick(ui_state.update)
  # the comma 4 draws at 60 Hz, so only the tizi's 20 Hz UI can pace to the model
  if HARDWARE.get_device_type() == 'tizi':
    gui_app.set_frame_start(ModelFrameClock().wait)
  if BIG_UI:
    MainLayout()
  else:
    MiciMainLayout()

  pm = messaging.PubMaster(['uiDebug'])
  for should_render, frame_time, cpu_time in gui_app.render():
    extra_start = time.monotonic()
    if not should_render:
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
