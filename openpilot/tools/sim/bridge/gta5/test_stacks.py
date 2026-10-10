"""Thread stacks for a hung bridge (gta5_stacks.py): on SIGUSR1, and when the loop's heartbeat stops."""
import faulthandler
import os
import signal
import sys
import time

from openpilot.tools.sim.bridge.gta5.gta5_stacks import Stall, stacks_path, watch


def test_stacks_path(monkeypatch):
  monkeypatch.delenv("GTA5_STACKS", raising=False)
  monkeypatch.setenv("GTA5_LOG", "/some/where/bridge.jsonl")
  assert stacks_path() == "/some/where/bridge_stacks.txt" and stacks_path("rx") == "/some/where/bridge_stacks_rx.txt"
  monkeypatch.setenv("GTA5_STACKS", "/x/s.txt")
  assert stacks_path("rx") == "/x/s_rx.txt"


def test_stacks_on_sigusr1(tmp_path, monkeypatch):
  monkeypatch.setenv("GTA5_STACKS", str(tmp_path / "stacks.txt"))
  f = watch("t")
  try:
    assert f is not None and f.name == str(tmp_path / "stacks_t.txt")
    os.kill(os.getpid(), signal.SIGUSR1)
    text = (tmp_path / "stacks_t.txt").read_text()
    assert f"kill -USR1 {os.getpid()}" in text and "test_stacks_on_sigusr1" in text
  finally:
    faulthandler.unregister(signal.SIGUSR1)
    faulthandler.enable(file=sys.__stderr__)
    f.close()


def test_stall_dumps_the_stacks_once_the_beat_stops(tmp_path, capsys):
  with open(tmp_path / "stacks.txt", "a", buffering=1) as f:
    stall = Stall(f, after=0.3, every=0.0)
    try:
      stall.beat(time.monotonic())
      time.sleep(0.8)  # holds the main thread, as a hang would
      assert "Timeout" in (tmp_path / "stacks.txt").read_text()
      stall.beat(time.monotonic())
      assert "the bridge loop stalled for" in capsys.readouterr().out
    finally:
      stall.stop()


def test_no_stall_while_beating(tmp_path, capsys):
  with open(tmp_path / "stacks.txt", "a", buffering=1) as f:
    stall = Stall(f, after=0.3, every=0.05)
    try:
      for _ in range(16):
        stall.beat(time.monotonic())
        time.sleep(0.05)
    finally:
      stall.stop()
  assert (tmp_path / "stacks.txt").read_text() == "" and capsys.readouterr().out == ""
