#!/usr/bin/env python3
"""Headless Slow Roads run: starts openpilot (launch_openpilot.sh) and the bridge, and prints engagement health every 6 s.
Run from the repository root with the venv active and OPENPILOT_PREFIX (and MODELD_DEV) exported.
Usage: probe_slowroads.py [seconds, default 90]; EXPERIMENTAL=1 for experimental mode."""
import os
import subprocess
import sys
import time
from multiprocessing import Queue
from pathlib import Path

from openpilot.cereal import messaging
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.tools.sim.bridge.slowroads.slowroads_bridge import SlowRoadsBridge

OpenpilotPrefix(os.environ["OPENPILOT_PREFIX"]).create_dirs()  # /dev/shm is cleared whenever the distro restarts
from openpilot.common.params import Params  # after create_dirs, which it needs

SIM_DIR = Path(__file__).parents[2]
SERVICES = ["selfdriveState", "onroadEvents", "modelV2", "cameraOdometry", "carState", "deviceMotion"]


def main(duration: float) -> None:
  Params().put_bool("ExperimentalMode", os.getenv("EXPERIMENTAL", "0") == "1")
  env = dict(os.environ, BLOCK="soundd", BIG="1", GALLIUM_DRIVER="d3d12")
  manager = subprocess.Popen("./launch_openpilot.sh", cwd=SIM_DIR, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  sm = messaging.SubMaster(SERVICES)
  time.sleep(8)  # let manager_init finish before the bridge writes params
  q: Queue = Queue()
  bridge = SlowRoadsBridge()
  bridge.test_run = False
  bridge_process = bridge.run(q)
  start = time.monotonic()
  last = 0.0
  active = 0
  try:
    while time.monotonic() - start < duration:
      sm.update(100)
      active += sm["selfdriveState"].active
      if time.monotonic() - last > 6:
        last = time.monotonic()
        m, cs = sm["modelV2"], sm["carState"]
        invalid = ",".join(s for s in SERVICES[2:] if not sm.valid[s]) or "none"
        print(f"t={last - start:4.0f} active={sm['selfdriveState'].active} n_active={active} exec={m.modelExecutionTime * 1000:.1f}ms " +
              f"drop={m.frameDropPerc:.0f}% vEgo={cs.vEgo:.1f} steer={cs.steeringAngleDeg:.1f} invalid={invalid} " +
              f"events={[e.name for e in sm['onroadEvents']]}", flush=True)
  finally:
    bridge.shutdown()
    bridge_process.terminate()
    manager.terminate()
    time.sleep(3)
    bridge_process.kill()
    manager.kill()
    while not q.empty():
      print("Q", q.get())
    os._exit(0)


if __name__ == "__main__":
  main(float(sys.argv[1]) if len(sys.argv) > 1 else 90)
