import sys
import time

if sys.platform == "win32" and sys.version_info < (3, 13):
  # the GetTickCount64 based clock only ticks every 15.6 ms; Python 3.13 moved monotonic to QueryPerformanceCounter
  # TODO: drop when the Python pin reaches 3.13
  setattr(time, "monotonic", time.perf_counter)  # noqa: B010 (a plain assignment is a type error for ty on Windows)
  setattr(time, "monotonic_ns", time.perf_counter_ns)  # noqa: B010
