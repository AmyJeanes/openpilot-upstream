import glob
import os
import time

from openpilot.common.test import OpenpilotTestCase
import openpilot.cereal.messaging as messaging
from openpilot.system.manager.process_config import managed_processes
from openpilot.common.hardware.hw import Paths
from openpilot.common.swaglog import cloudlog, ipchandler


class TestLogmessaged(OpenpilotTestCase):
  def setup_method(self):
    # clear the IPC buffer in case some other tests used cloudlog and filled it
    ipchandler.close()
    ipchandler.connect()

    managed_processes['logmessaged'].start()
    self.sock = messaging.sub_sock("logMessage", timeout=1000, conflate=False)
    self.error_sock = messaging.sub_sock("logMessage", timeout=1000, conflate=False)

    # ensure sockets are connected and the daemon is up (spawning it takes a while on Windows)
    for _ in range(100):
      cloudlog.error("logmessaged ready check")
      time.sleep(0.1)
      if messaging.drain_sock(self.sock):
        break
    time.sleep(0.5)
    messaging.drain_sock(self.sock)
    messaging.drain_sock(self.error_sock)

  def teardown_method(self):
    del self.sock
    del self.error_sock
    managed_processes['logmessaged'].stop(block=True)

  def _get_log_files(self):
    return list(glob.glob(os.path.join(Paths.swaglog_root(), "swaglog.*")))

  def _log_size(self):
    return sum(os.path.getsize(f) for f in self._get_log_files())

  def test_simple_log(self):
    msgs = [f"abc {i}" for i in range(10)]
    for m in msgs:
      cloudlog.error(m)
    time.sleep(0.5)
    m = messaging.drain_sock(self.sock)
    assert len(m) == len(msgs)
    assert len(self._get_log_files()) >= 1

  def test_big_log(self):
    n = 10
    msg = "a"*3*1024*1024
    base = self._log_size()  # the ready checks from setup_method are in the same files
    for _ in range(n):
      cloudlog.info(msg)
    for _ in range(300):  # writing 30 MB takes a while on a loaded machine
      time.sleep(0.1)
      logsize = self._log_size() - base
      if logsize > n * len(msg):
        break

    msgs = messaging.drain_sock(self.sock)
    assert len(msgs) == 0

    assert (n*len(msg)) < logsize < (n*(len(msg)+1024))
