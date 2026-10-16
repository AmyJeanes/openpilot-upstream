"""GTA5_NAV_PATH=navd: comma navd's requests (navInstruction.request) as the simulated driver's stalk (NavdRequests)."""
from types import SimpleNamespace

from openpilot.tools.sim.bridge.gta5.gta5_navd import NavdRequests


class FakeSM:
  def __init__(self):
    self.req = SimpleNamespace(kind="none", side="none", reason="other", distance=0.0, laneChanges=0, stage=3, id=0, approved=False)

  def update(self, timeout=0):
    pass

  def __getitem__(self, key):
    return SimpleNamespace(request=self.req)

  def set(self, **kw):
    self.req = SimpleNamespace(**{**vars(self.req), **kw})


def test_requests_work_the_stalk():
  sm = FakeSM()
  nr = NavdRequests(sm)
  assert nr.step().requests == []
  sm.set(kind="laneChange", side="right", id=1, laneChanges=2, reason="turn")
  out = nr.step()
  assert out.requests == ["laneChangeRight"] and out.request["laneChanges"] == 2 and not nr.signaling
  assert nr.step().requests == []  # the same request: nothing more
  sm.set(kind="none", side="none")
  assert nr.step().requests == ["cancelSignal"]
  sm.set(kind="laneChange", side="right", id=2, laneChanges=1)
  assert nr.step().requests == ["laneChangeRight"]
  sm.set(kind="turn", side="right", id=3)  # the turn's signal takes over: off and on again, as the planner does
  assert nr.step().requests == ["cancelSignal", "signalTurnRight"] and nr.signaling
  sm.set(kind="turn", side="left", id=4)
  assert nr.step().requests == ["cancelSignal", "signalTurnLeft"]
