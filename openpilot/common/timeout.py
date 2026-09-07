import signal
import sys
import threading

class TimeoutException(Exception):
  pass

class Timeout:
  """
  Timeout context manager.
  For example this code will raise a TimeoutException:
  with Timeout(seconds=5, error_msg="Sleep was too long"):
    time.sleep(10)

  On Windows the timeout interrupts Python code and sleeps, but not a blocking wait on a
  child process or pipe: that only raises once the wait itself returns.
  """
  def __init__(self, seconds, error_msg=None):
    if error_msg is None:
      error_msg = f'Timed out after {seconds} seconds'
    self.seconds = seconds
    self.error_msg = error_msg

  def handle_timeout(self, signume, frame):
    raise TimeoutException(self.error_msg)

  def __enter__(self):
    if sys.platform == "win32":
      # no SIGALRM: a timer thread raises SIGINT, which becomes a KeyboardInterrupt in the main thread and
      # also wakes time.sleep (interrupt_main would not); __exit__ translates it
      self.expired = False
      self.timer = threading.Timer(self.seconds, self._interrupt)
      self.timer.start()
      return
    signal.signal(signal.SIGALRM, self.handle_timeout)
    signal.alarm(self.seconds)

  def _interrupt(self):
    self.expired = True
    signal.raise_signal(signal.SIGINT)

  def __exit__(self, exc_type, exc_val, exc_tb):
    if sys.platform == "win32":
      self.timer.cancel()
      if self.expired and exc_type is KeyboardInterrupt:
        raise TimeoutException(self.error_msg) from None
      return
    signal.alarm(0)
