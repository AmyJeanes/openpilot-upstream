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

FRAME_CLOCK_TIMEOUT_MS = 50  # one camera period


class DriverMonitoringClock:
  """Presents onroad frames just after driver monitoring, the last GPU job in each camera period.

  While clocked, each frame waits for the next camera frame's model output before drawing, and for that frame's DM
  output before presenting. Output for an earlier frame can still be queued; it's skipped so the UI can't lock a frame out of step.
  """
  def __init__(self):
    self._dm = messaging.sub_sock('driverStateV2', conflate=True, timeout=FRAME_CLOCK_TIMEOUT_MS)
    self._model = messaging.sub_sock('modelV2', conflate=True, timeout=FRAME_CLOCK_TIMEOUT_MS)
    self._dm_running = False
    self._model_running = False
    self._frame_id = 0

  def wait_for_model(self):
    model = messaging.recv_one_or_none(self._model)
    if (model is None or model.modelV2.frameId <= self._frame_id) and self._model_running:
      model = messaging.recv_one(self._model)
    self._model_running = model is not None
    if model is not None:
      self._frame_id = model.modelV2.frameId
    ui_state.update()

  def wait(self) -> bool:
    if not ui_state.started:
      return False
    dm = messaging.recv_one_or_none(self._dm)
    # without DM keep the fixed rate rather than waiting out a timeout every frame
    if (dm is None or dm.driverStateV2.frameId < self._frame_id) and self._dm_running:
      dm = messaging.recv_one(self._dm)
    self._dm_running = dm is not None
    if dm is not None:
      self._frame_id = dm.driverStateV2.frameId
    return self._dm_running


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
    clock = DriverMonitoringClock()
    gui_app.set_frame_clock(clock.wait, frame_start=clock.wait_for_model)

  pm = messaging.PubMaster(['uiDebug'])
  for should_render, frame_time, cpu_time in gui_app.render():
    extra_start = time.monotonic()
    # while clocked, the clock's frame start has already updated the state for this frame
    if not (should_render and gui_app.frame_clocked):
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
