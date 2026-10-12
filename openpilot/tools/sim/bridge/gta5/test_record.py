"""The recorder's segments as frames come (gta5_record.py), with stand-in encoders."""
from types import SimpleNamespace

from openpilot.tools.sim.bridge.gta5 import gta5_record
from openpilot.tools.sim.bridge.gta5.gta5_record import Recorder


class Encoder:
  def __init__(self):
    self.ok = True
    self.thread = SimpleNamespace(is_alive=lambda: self.ok)

  def add(self, frame) -> bool:
    return self.ok

  def finish(self):
    pass


class Segment:
  def __init__(self, path, info, size):
    self.road, self.wide = Encoder(), Encoder()
    self.rows: list = []
    self.md, self.mount, self.broken, self.done_at = None, None, "", 0.0


def test_recording_only_while_a_segment_takes_frames(tmp_path, monkeypatch):
  monkeypatch.setattr(gta5_record, "Segment", Segment)
  monkeypatch.setattr(gta5_record, "MODEL_WAIT", 1e9)  # nothing saved
  rec = Recorder(str(tmp_path))
  rec._row = lambda *args: []
  car = {"inVehicle": True, "pos": [0.0, 0.0, 0.0], "resets": 0}
  t = iter(range(100))

  def frame(**state):
    rec.add(False, b"frame", {**car, **state}, next(t) * 0.05)
    return rec.recording

  assert not rec.recording  # idle
  assert frame()
  assert not frame(inVehicle=False) and rec.segment is None  # on foot
  assert frame()
  assert not frame(paused=True)
  n = rec.segment_count
  car["resets"] = 1
  assert frame() and rec.segment_count == n + 1  # a teleport: the next segment starts at once
  rec.segment.road.ok = False
  assert not frame() and rec.failed  # the encoder failed
  assert not frame() and rec.segment is None
