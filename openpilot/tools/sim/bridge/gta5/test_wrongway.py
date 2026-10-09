#!/usr/bin/env python3
"""Offline checks of the wrong-way clips (gta5_wrongway.py) and expert mode driving them, on a simulated car along a
straight undivided road: run as a script."""
import math

import numpy as np

from openpilot.tools.sim.bridge.gta5.gta5_expert import Expert
from openpilot.tools.sim.bridge.gta5.gta5_wrongway import WrongWay, suitable
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import BACKWARD, FORWARD, Lane, RouteLanes, Section, Span
from openpilot.tools.sim.bridge.gta5.map.router import Route

W = 5.5


def road(lanes: int = 1, length: float = 800.0, median: float = 0.0, junction: float | None = None) -> Route:
  """A straight road north from the origin, its line between the directions, `lanes` each way."""
  pts = np.stack([np.zeros(int(length / 20) + 1), np.arange(0.0, length + 1, 20.0)], axis=1)
  ours = [Span(Lane(FORWARD, W), median / 2 + k * W, median / 2 + (k + 1) * W, 1) for k in range(lanes)]
  onc = [Span(Lane(BACKWARD, W), -(median / 2 + (k + 1) * W), -(median / 2 + k * W), -1) for k in reversed(range(lanes))]
  sec = Section(onc + ours, (onc[0].left, ours[-1].right))
  r = Route(pts, [{"type": 1, "begin_shape_index": 0}])
  r.osm_lanes = True
  r._lanes = RouteLanes(pts, [sec] * (len(pts) - 1))
  if junction is not None:
    r.junctions = [junction]
  return r


class Car:
  def __init__(self, route: Route, lane: int = 0):
    self.r = route
    self.x, self.y = route.section(0).offset(lane), 0.0
    self.h, self.v, self.t, self.collisions = 0.0, 0.0, 0.0, 0  # game heading: counterclockwise from north

  def state(self) -> dict:
    self.r.locate(np.array([self.x, self.y]), heading=self.h)
    sec = self.r.section(self.r.seg)
    lane = [sec.lane(self.r.right), sec.lanes]  # as the plugin reads it, whatever the car's heading
    return {"pos": [self.x, self.y, 0.0], "vEgo": self.v, "heading": self.h, "lane": lane, "lanePlugin": lane, "laneMap": None,
            "t": self.t, "collisions": self.collisions, "ai": {"on": False}}

  def move(self, curvature: float, accel: float, dt: float = 0.05):
    self.v = max(self.v + accel * dt, 0.0)
    self.h += math.degrees(curvature * self.v * dt)
    self.x += -math.sin(math.radians(self.h)) * self.v * dt
    self.y += math.cos(math.radians(self.h)) * self.v * dt
    self.t += dt

  def ai_drive(self, lane: float, speed: float, dt: float = 0.05):
    """Stands in for the game's AI: back towards a lane's centre."""
    target = self.r.section(0).offset(lane)
    h = math.radians(self.h)
    dx, dy = target - self.x, 15.0
    left = dx * -math.cos(h) + dy * -math.sin(h)
    self.move(2 * left / (dx * dx + dy * dy), 0.8 * (speed - self.v), dt)


def drive(cfg: dict, route: Route, lane: int = 0, seconds: float = 90.0, collide_at: float | None = None):
  car = Car(route, lane)
  ww = WrongWay(cfg)
  phases, lanes, max_dev = [], [], 0.0
  while car.t < seconds and ww.finished is None:
    s = car.state()
    if collide_at is not None and car.y > collide_at:
      car.collisions = 1
    msg = ww.step(route, s, car.t, car.collisions)
    if not phases or phases[-1] != ww.phase:
      phases.append(ww.phase)
    if s["lane"]:
      lanes.append(s["lane"][0])
    if ww.phase in ("hold",):
      max_dev = max(max_dev, ww.dev)
    if msg is None:
      car.ai_drive(ww.from_lane, cfg.get("speed", 10.0))
    else:
      car.move(msg["curvature"], msg["accel"])
  return ww, car, phases, lanes, max_dev


def test_path_clip():
  for lanes, target in ((1, -1.0), (1, -0.5), (2, -2.0)):
    for speed, drift in ((6.0, 12.0), (10.0, 70.0), (14.0, 30.0)):
      cfg = {"clip": "t", "lane": target, "speed": speed, "drift_m": drift, "hold_s": 3.0, "recover": "path", "recover_m": 50.0,
             "from_lane": lanes - 1}
      ww, car, phases, seen, dev = drive(cfg, road(lanes), lane=lanes - 1)
      assert ww.finished == "done", (lanes, target, speed, ww.finished, ww.why)
      assert phases == ["approach", "drift", "hold", "recover", "done"], phases
      if target <= -1:
        assert min(seen) <= target, (lanes, target, speed, min(seen))
      assert car.r.lane()[0] == lanes - 1, car.r.lane()
      assert dev < 1.5, (lanes, target, speed, dev)  # a swerve overshoots into the hold a little
      assert car.v < 0.5
  print("path clips: ok")


def test_ai_clip():
  cfg = {"clip": "t", "lane": -1, "speed": 10.0, "drift_m": 60.0, "hold_s": 3.0, "recover": "ai"}
  ww, car, phases, seen, _ = drive(cfg, road(1))
  assert phases == ["approach", "drift", "hold", "ai_recover", "done"], phases
  assert ww.finished == "done" and min(seen) == -1
  rec = list(ww.events)  # taken by nobody here: all of them
  assert any(e.get("phase") == "done" and e.get("recovered_s") is not None for e in rec), rec
  print("ai clip: ok")


def test_aborts():
  cfg = {"clip": "t", "lane": -1, "speed": 10.0, "drift_m": 60.0, "hold_s": 3.0, "recover": "path"}
  ww, car, phases, _, _ = drive(cfg, road(1), collide_at=150.0)
  assert ww.finished == "abort: collision" and phases[-1] == "abort" and car.v < 0.5, (ww.finished, phases)
  for r, want in ((road(1, median=4.0), "median"), (road(1, junction=150.0), "junction"), (road(3), "3+3 lanes"),
                  (road(1, length=200.0), "too short")):
    ww, car, phases, _, _ = drive(cfg, r)
    assert ww.finished is not None and want in ww.finished, (want, ww.finished)
    assert phases == ["abort"]
  print("aborts: ok")


def test_suitable():
  assert suitable(road(1), 100, 300) is None
  assert suitable(road(2), 100, 300) is None
  assert "median" in suitable(road(2, median=3.0), 100, 300)
  print("suitable: ok")


def test_expert():
  """Expert mode with a clip: the AI off and the controller's controls, then the AI on for the recovery."""
  sent = []
  e = Expert(sent.append, lambda: None, lambda: None)
  e.cfg = {**e.cfg, "on": True, "wrongway": {"clip": "t", "lane": -1, "speed": 10.0, "drift_m": 60.0, "hold_s": 2.0}}
  e.on = True
  e.ww = e._wrongway(e.cfg["wrongway"])
  route = road(1)
  car = Car(route)
  ai_on_at = None
  for _ in range(4000):
    s = car.state()
    assert e.update(s, route, False)
    ctl = [m for m in sent if m.get("type") == "control"]
    if e.ww.phase == "approach":
      assert not e.ai_drives and e.ww_driving
    if ai_on_at is None and any(m.get("type") == "ai" and m.get("on") == 1 for m in sent):
      ai_on_at = e.ww.phase
    if e.ww_driving:
      car.move(ctl[-1]["curvature"], ctl[-1]["accel"])
    else:
      car.ai_drive(0, 10.0)
    if e.ww.finished:
      break
  assert sent[0] == {"type": "ai", "on": 0, "indicator": "off"}, sent[0]
  assert ai_on_at == "ai_recover", ai_on_at
  assert e.ai_drives and e.ww.finished == "done"
  print("expert: ok")


def test_record_rows():
  """gta5_record's per-frame wrongway row: the clip and phase while one runs, nothing (-1, 0) otherwise."""
  from types import SimpleNamespace
  from openpilot.tools.sim.bridge.gta5.gta5_record import WW_COLUMNS, Recorder
  from openpilot.tools.sim.bridge.gta5.gta5_wrongway import PHASES
  rec = Recorder.__new__(Recorder)
  rec.segment = SimpleNamespace(ww_clips=[])
  e = Expert(lambda m: None, lambda: None, lambda: None)
  route = road(1)
  assert rec._ww_row(e, route, {})[:2] == [-1, 0] and not rec.segment.ww_clips
  e.active, e.ww = True, WrongWay({"clip": "c1", "lane": -1})
  car = Car(route)
  s = car.state()
  e.ww_driving = e.ww.step(route, s, 0.0, 0) is not None
  row = rec._ww_row(e, route, s)
  assert len(row) == len(WW_COLUMNS) and row[0] == 0 and PHASES[row[1]] == "approach" and row[2] == 0.0, row
  assert rec.segment.ww_clips[0]["clip"] == "c1" and rec.segment.ww_clips[0]["path"]
  print("record rows: ok")


if __name__ == "__main__":
  test_suitable()
  test_path_clip()
  test_ai_clip()
  test_aborts()
  test_expert()
  test_record_rows()
