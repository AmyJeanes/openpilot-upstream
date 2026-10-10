"""The threads' stacks of a bridge process, for a hang, through faulthandler: it needs neither a debugger (py-spy can't
attach under WSL's ptrace_scope 1 without sudo) nor the GIL, so it works while another thread holds that.

watch() sets a process up: `kill -USR1 <pid>` appends its threads' stacks to its file (GTA5_STACKS, else
bridge_stacks.txt beside GTA5_LOG, else in /tmp; a named process's file has its name added, bridge_stacks_rx.txt), and a
fatal signal (a segfault, a bus error) prints them to stderr, the bridge's log, as the process dies. Stall appends them
when the bridge loop's heartbeat stops for a while."""
import faulthandler
import os
import signal
import sys
import time
from typing import TextIO

STALL = 20.0  # s without a heartbeat
BEAT_EVERY = 1.0  # s


def stacks_path(name: str = "") -> str:
  path = os.getenv("GTA5_STACKS")
  if not path:
    log = os.getenv("GTA5_LOG")
    path = os.path.join(os.path.dirname(os.path.abspath(log)) if log else "/tmp", "bridge_stacks.txt")
  root, ext = os.path.splitext(path)
  return f"{root}_{name}{ext}" if name else path


def watch(name: str = "") -> TextIO | None:
  """This process's stacks on SIGUSR1 and fatal signals; its stacks file, None where that can't be opened."""
  try:
    faulthandler.enable(file=sys.stderr, all_threads=True)
  except (AttributeError, OSError, ValueError):  # a stderr with no file descriptor
    pass
  path = stacks_path(name)
  try:
    f = open(path, "a", buffering=1)
  except OSError as e:
    print(f"gta5: no thread stacks on SIGUSR1 ({path}: {e})", flush=True)
    return None
  pid = os.getpid()
  f.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} pid {pid}{' (' + name + ')' if name else ''}: kill -USR1 {pid} adds its stacks\n")
  faulthandler.register(signal.SIGUSR1, file=f, all_threads=True)
  return f


class Stall:
  """Appends the threads' stacks to `file` once beat() hasn't come for `after` s: faulthandler's timer runs in a C thread
  of its own, so it fires even while a thread holds the GIL. beat() rearms it at most every `every` s, as each rearm
  starts that thread afresh."""

  def __init__(self, file: TextIO, after: float = STALL, every: float = BEAT_EVERY):
    self.file = file
    self.after = after
    self.every = every
    self.last: float | None = None
    self.armed = -1e9

  def beat(self, now: float):
    if self.last is not None and now - self.armed > self.after:  # the timer went off
      print(f"gta5: the bridge loop stalled for {now - self.last:.0f} s; the threads' stacks are in {self.file.name}", flush=True)
    self.last = now
    if now - self.armed >= self.every:
      self.armed = now
      faulthandler.dump_traceback_later(self.after, file=self.file)

  def stop(self):
    faulthandler.cancel_dump_traceback_later()
