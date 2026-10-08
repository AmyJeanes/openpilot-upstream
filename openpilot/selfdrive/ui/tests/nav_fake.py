#!/usr/bin/env python3
"""Fake navigation for the onroad UI's nav card, on a test prefix: navInstruction and navRoute for a made-up route (north,
a right turn, on to the destination), and with --onroad the openpilot state the onroad view needs (started, engaged, a
speed) and a still camera frame. The UI run separately on the same OPENPILOT_PREFIX shows it:

  OPENPILOT_PREFIX=uiport python selfdrive/ui/tests/nav_fake.py --onroad --scene approach
  OPENPILOT_PREFIX=uiport BIG=1 SCALE=1 python selfdrive/ui/ui.py

Scenes: none (no route), cruise (the turn far ahead), approach (near it, with the lanes), turn, arrive, drive (the car
drives the route from the start, through the turn to the destination, at a city speed)."""
import argparse
import math
import os
import sys
import time

import numpy as np

PREFIX = os.environ.get("OPENPILOT_PREFIX", "")
if __name__ == "__main__" and (not PREFIX or PREFIX == "gta5"):
  sys.exit("set OPENPILOT_PREFIX to a test prefix of its own (not the live gta5 one)")

from openpilot.cereal import log, messaging

LAT0, LON0 = 34.0, -118.3
M_PER_DEG = 111319.49
TURN_AT, AFTER = 2614.0, 1600.0  # m north to the turn, then m east to the destination
TRIP_SPEED = 7.5  # m/s, for the time left
ROADS = {"short": "Vespucci Blvd", "long": "Hillcrest Ridge Access Rd"}
DESTINATION = "Del Perro Pier"
# scene: m along the route, the car's speed (m/s)
SCENES = {"cruise": (TURN_AT - 2414.0, 14.75), "approach": (TURN_AT - 183.0, 12.5), "turn": (TURN_AT - 10.0, 6.26),
          "arrive": (TURN_AT + AFTER - 120.0, 8.0)}
DRIVE_SPEED = 12.0


def to_lat_lon(x: float, y: float) -> tuple[float, float]:
  return float(LAT0 + y / M_PER_DEG), float(LON0 + x / (M_PER_DEG * math.cos(math.radians(LAT0))))


def route_points(after: float = AFTER, variant: int = 0) -> np.ndarray:
  """variant > 0: the same route as a router sends it again (a reroute), with a point more along its way."""
  extra = [(0.0, TURN_AT * variant / (variant + 1))] if variant else []
  return np.array([(0.0, 0.0), *extra, (0.0, TURN_AT), (after, TURN_AT)])


def pose_at(s: float) -> tuple[tuple[float, float], float]:
  """The car s m along the route, and its heading (deg clockwise from north)."""
  if s < TURN_AT:
    return (0.0, s), 0.0
  return (s - TURN_AT, TURN_AT), 90.0


def nearby_roads() -> list[tuple[np.ndarray, float]]:
  """A street grid around the route, for the map."""
  roads = []
  for x in np.arange(-750.0, AFTER + 751.0, 250.0):
    roads.append((np.array([(x, -400.0), (x, TURN_AT + 750.0)]), 14.0 if x == 0 else 9.0))
  for y in np.arange(-250.0, TURN_AT + 751.0, 250.0):
    roads.append((np.array([(-750.0, y), (AFTER + 750.0, y)]), 9.0))
  roads.append((np.array([(-750.0, TURN_AT), (AFTER + 750.0, TURN_AT)]), 14.0))
  return roads


def route_message(after: float = AFTER, variant: int = 0):
  msg = messaging.new_message("navRoute", valid=True)
  pts = route_points(after, variant)
  coords = msg.navRoute.init("coordinates", len(pts))
  for c, (x, y) in zip(coords, pts, strict=True):
    c.latitude, c.longitude = to_lat_lon(x, y)
  roads = nearby_roads()
  out = msg.navRoute.init("roads", len(roads))
  for o, (line, width) in zip(out, roads, strict=True):
    cs = o.init("coordinates", len(line))
    for c, (x, y) in zip(cs, line, strict=True):
      c.latitude, c.longitude = to_lat_lon(x, y)
    o.width = width
  return msg


def instruction_message(s: float | None, road: str = "short", lanes: bool = True, after: float = AFTER):
  """navInstruction for the car s m along the route; None for no route."""
  msg = messaging.new_message("navInstruction", valid=True)
  ni = msg.navInstruction
  if s is None:
    return msg
  ni.valid = True
  ni.destinationName = DESTINATION
  end = TURN_AT + after
  ahead = [(TURN_AT, "turn", "right", ROADS[road]), (end, "arrive", "straight", "")]
  ahead = [m for m in ahead if m[0] > s]
  if ahead:
    at, kind, modifier, primary = ahead[0]
    ni.maneuverType, ni.maneuverModifier, ni.maneuverPrimaryText, ni.maneuverDistance = kind, modifier, primary, at - s
    mans = ni.init("allManeuvers", len(ahead))
    for out, (a, k, mod, p) in zip(mans, ahead, strict=True):
      out.distance, out.type, out.modifier, out.primaryText = a - s, k, mod, p
  ni.distanceRemaining = max(end - s, 0.0)
  ni.timeRemaining = ni.timeRemainingTypical = ni.distanceRemaining / TRIP_SPEED
  (x, y), bearing = pose_at(s)
  ni.position.latitude, ni.position.longitude = to_lat_lon(x, y)
  ni.bearingDeg = bearing
  # four lanes our way (two oncoming, which the card leaves off), the right two for the turn, the car in the second
  if lanes and s < TURN_AT and TURN_AT - s < 400.0:
    ni.showFull = True
    spec = [(["straight"], False, "none", True, False), (["straight"], False, "none", True, False),
            (["straight"], False, "none", False, False), (["straight"], False, "none", False, True),
            (["straight", "right"], True, "right", False, False), (["right"], True, "right", False, False)]
    out_lanes = ni.init("lanes", len(spec))
    for o, (dirs, active, use, oncoming, current) in zip(out_lanes, spec, strict=True):
      o.directions, o.active, o.activeDirection, o.oncoming, o.current = dirs, active, use, oncoming, current
    ni.laneDistance = max(TURN_AT - s - 60.0, 0.0)  # the car isn't in a turn lane yet
  return msg


class FakeOnroad:
  """The state the onroad view reads: started, engaged (or not) at a speed, Experimental mode, an alert if given."""
  SERVICES = ["deviceState", "pandaStates", "selfdriveState", "carState", "controlsState", "carParams", "deviceMotion"]

  def __init__(self):
    self.pm = messaging.PubMaster(self.SERVICES)
    self.v = 12.0
    self.engaged = True
    self.experimental = True
    self.alert: tuple[str, str, str] | None = None  # text1, text2, size

  def send(self):
    msgs = {s: messaging.new_message(s, valid=True) for s in self.SERVICES if s != "pandaStates"}
    msgs["deviceState"].deviceState.started = True
    msgs["deviceState"].deviceState.deviceType = "tizi"
    ps = messaging.new_message("pandaStates", 1, valid=True)
    ps.pandaStates[0].ignitionLine = True
    ps.pandaStates[0].pandaType = log.PandaState.PandaType.tres
    msgs["pandaStates"] = ps
    ss = msgs["selfdriveState"].selfdriveState
    ss.enabled = ss.active = self.engaged
    ss.state = log.SelfdriveState.OpenpilotState.enabled if self.engaged else log.SelfdriveState.OpenpilotState.disabled
    ss.experimentalMode = self.experimental
    if self.alert:
      ss.alertText1, ss.alertText2, ss.alertSize = self.alert
      ss.alertStatus = log.SelfdriveState.AlertStatus.normal
    cs = msgs["carState"].carState
    cs.vEgo, cs.vEgoCluster, cs.vCruiseCluster, cs.cruiseState.enabled = self.v, self.v, 56.0, True
    msgs["controlsState"].controlsState.deprecated.vCruise = 56.0
    msgs["carParams"].carParams.openpilotLongitudinalControl = True
    for s, m in msgs.items():
      self.pm.send(s, m)


class FakeCamera:
  """A still frame as the road camera's stream."""
  W, H = 1928, 1208

  def __init__(self, path: str):
    from msgq.visionipc import VisionIpcServer
    from openpilot.cereal.visionipc import VisionStreamType
    self.stream = VisionStreamType.VISION_STREAM_NARROW_ROAD
    self.vipc = VisionIpcServer("camerad")
    for stream in (VisionStreamType.VISION_STREAM_NARROW_ROAD, VisionStreamType.VISION_STREAM_WIDE_ROAD):
      self.vipc.create_buffers_with_sizes(stream, 4, self.W, self.H, self.W * self.H * 3 // 2, self.W, self.W * self.H)
    self.vipc.start_listener()
    self.frame = nv12(path, self.W, self.H)
    self.frame_id = 0

  def send(self):
    t = self.frame_id * 50_000_000
    self.vipc.send(self.stream, self.frame, self.frame_id, t, t)
    self.frame_id += 1


def nv12(path: str, w: int, h: int) -> bytes:
  from PIL import Image
  im = np.asarray(Image.open(path).convert("RGB").resize((w, h)), np.float32)
  r, g, b = im[..., 0], im[..., 1], im[..., 2]
  y = 0.257 * r + 0.504 * g + 0.098 * b + 16
  u = -0.148 * r - 0.291 * g + 0.439 * b + 128
  v = 0.439 * r - 0.368 * g - 0.071 * b + 128
  uv = np.stack([u[::2, ::2], v[::2, ::2]], axis=-1)
  return np.concatenate([y.clip(0, 255).astype(np.uint8).ravel(), uv.clip(0, 255).astype(np.uint8).ravel()]).tobytes()


class FakeNav:
  """navInstruction at 10 Hz and navRoute every 2 s, for a scene."""
  def __init__(self, scene: str = "cruise", road: str = "short", drive_from: float = 0.0):
    self.pm = messaging.PubMaster(["navInstruction", "navRoute"])
    self.scene, self.road = scene, road
    self.drive_s = drive_from
    self._route_t = -1e9
    self.after, self.variant = AFTER, 0  # the destination, m on from the turn; reroutes so far

  def reroute(self):
    self.variant += 1
    self._route_t = -1e9

  def new_destination(self):
    self.after += 400.0
    self.variant = 0
    self._route_t = -1e9

  def position(self) -> tuple[float | None, float]:
    """(m along, or None for no route; the car's speed)."""
    if self.scene == "none":
      return None, 12.0
    if self.scene == "drive":
      return self.drive_s, DRIVE_SPEED
    return SCENES[self.scene]

  def send(self, now: float, dt: float = 0.0):
    if self.scene == "drive":
      self.drive_s = (self.drive_s + DRIVE_SPEED * dt) % (TURN_AT + self.after)
    s, _ = self.position()
    if s is not None and now - self._route_t > 2.0:
      self.pm.send("navRoute", route_message(self.after, self.variant))
      self._route_t = now
    if s is None:
      self._route_t = -1e9
    self.pm.send("navInstruction", instruction_message(s, self.road, after=self.after))


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--scene", default="drive", choices=["none", *SCENES, "drive"])
  ap.add_argument("--road", default="short", choices=list(ROADS))
  ap.add_argument("--onroad", action="store_true", help="also the openpilot state the onroad view needs")
  ap.add_argument("--camera", help="a still frame (png) as the road camera, with --onroad")
  args = ap.parse_args()
  os.makedirs(f"/dev/shm/msgq_{PREFIX}", exist_ok=True)
  nav = FakeNav(args.scene, args.road)
  op = FakeOnroad() if args.onroad else None
  cam = FakeCamera(args.camera) if args.onroad and args.camera else None
  last = time.monotonic()
  while True:
    now = time.monotonic()
    nav.send(now, now - last)
    last = now
    if op is not None:
      op.v = nav.position()[1]
      op.send()
    if cam is not None:
      cam.send()
    time.sleep(0.05)


if __name__ == "__main__":
  main()
