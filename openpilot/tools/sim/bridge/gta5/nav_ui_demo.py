#!/usr/bin/env python3
"""The onroad UI's navigation view, offline: a route over the map's roads (gta5.osm.pbf, routed here by shortest path,
no Valhalla), the bridge's nav messages for a car placed along it (gta5_nav_msgs.py), and the openpilot state the onroad
view needs, all published under OPENPILOT_PREFIX to the UI run in this process, which saves screenshots of each scene.
The car drives along the route in real time in each scene, reporting its own heading and a yaw rate (deviceMotion) as a
car would; the drive scene follows it through the first turn and checks the map turns with it smoothly.

  OPENPILOT_PREFIX=navui BIG=1 SCALE=0.5 GALLIUM_DRIVER=d3d12 python nav_ui_demo.py --out <dir> [--find]

--find lists candidate routes (start and destination nodes) with turns and lane guidance, for --start/--dest."""
import argparse
import heapq
import math
import os
import sys
import time

import numpy as np

PREFIX = os.environ.get("OPENPILOT_PREFIX", "")
if not PREFIX or PREFIX == "gta5":
  sys.exit("set OPENPILOT_PREFIX to a test prefix of its own (not the live gta5)")
os.makedirs(f"/dev/shm/msgq_{PREFIX}", exist_ok=True)

from openpilot.cereal import log, messaging
from openpilot.selfdrive.navd import lane_slots
from openpilot.tools.sim.bridge.gta5.gta5_nav_msgs import NavMessages, heading_at, point_at
from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import OsmLanes, oneway_of
from openpilot.tools.sim.bridge.gta5.map.router import Route

MAP = os.environ.get("GTA5_MAP", os.path.expanduser("~/gta5map_lanes"))
CAMERA = os.path.expanduser("~/gta5test/light1/s1_road.png")
TURN_MIN = 30.0  # deg through a junction that makes a maneuver
SPEED = 12.0  # m/s, for the route's times
CITY = (-400.0, 1200.0, -2000.0, -400.0)  # x0, x1, y0, y1: where --find picks routes


def shortest(osm: OsmLanes, a: int, b: int) -> list[int] | None:
  """Dijkstra over the map's ways, one-way streets their way only."""
  dist, prev, todo = {a: 0.0}, {}, [(0.0, a)]
  while todo:
    d, n = heapq.heappop(todo)
    if n == b:
      break
    if d > dist.get(n, math.inf):
      continue
    for m in osm.links.get(n, []):
      wid, along = osm.pairs[(n, m)]
      tags = osm.ways[wid][0]
      ow = oneway_of(tags)
      if ow == 1 and not along or ow == -1 and along or tags.get('highway') == 'service':
        continue
      nd = d + float(np.hypot(*(osm.node_xy(m) - osm.node_xy(n))))
      if nd < dist.get(m, math.inf):
        dist[m], prev[m] = nd, n
        heapq.heappush(todo, (nd, m))
  if b not in prev:
    return None
  path = [b]
  while path[-1] != a:
    path.append(prev[path[-1]])
  return path[::-1]


def route_of(osm: OsmLanes, nodes: list[int]) -> Route:
  """A Route through the nodes, with Valhalla-like maneuvers at the junctions it turns at."""
  pts = np.array([osm.node_xy(n) for n in nodes], float)
  along = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))))
  names = [osm.ways[osm.pairs[(a, b)][0]][0].get('name', '') for a, b in zip(nodes, nodes[1:], strict=False)]

  def heading(s0, s1):
    p0 = [np.interp(s0, along, pts[:, 0]), np.interp(s0, along, pts[:, 1])]
    p1 = [np.interp(s1, along, pts[:, 0]), np.interp(s1, along, pts[:, 1])]
    return math.degrees(math.atan2(p1[0] - p0[0], p1[1] - p0[1]))

  mans = [{'type': 1, 'begin_shape_index': 0, 'street_names': [names[0]] if names[0] else []}]
  for i in range(1, len(nodes) - 1):
    if osm.degree.get(nodes[i], 0) < 2 or len(osm.links.get(nodes[i], [])) < 3:
      continue
    s = along[i]
    turn = (heading(s, s + 25) - heading(s - 25, s) + 180) % 360 - 180
    if abs(turn) < TURN_MIN or along[i] - along[mans[-1]['begin_shape_index']] < 30:
      continue
    kind = (9 if abs(turn) < 45 else 10 if abs(turn) < 135 else 11) if turn > 0 else (16 if abs(turn) < 45 else 15 if abs(turn) < 135 else 14)
    mans.append({'type': kind, 'begin_shape_index': i, 'street_names': [names[i]] if names[i] else []})
  mans.append({'type': 4, 'begin_shape_index': len(nodes) - 1, 'street_names': []})
  for m, nxt in zip(mans, mans[1:] + [None], strict=True):
    end = along[nxt['begin_shape_index']] if nxt else along[-1]
    m['length'], m['time'] = (end - along[m['begin_shape_index']]) / 1000, (end - along[m['begin_shape_index']]) / SPEED
  return Route(pts, mans, osm=osm)


def load_map() -> OsmLanes:
  print(f"loading {MAP}/gta5.osm.pbf", flush=True)
  return OsmLanes.load(os.path.join(MAP, "gta5.osm.pbf"), lambda lat, lon: to_game(lat, lon))


def find(osm: OsmLanes, n: int = 40, seed: int = 1):
  rng = np.random.default_rng(seed)
  x0, x1, y0, y1 = CITY
  ids = [k for k, (x, y) in zip(osm.ids, osm.xy, strict=True) if x0 < x < x1 and y0 < y < y1 and k in osm.links]
  for _ in range(n):
    a, b = (int(v) for v in rng.choice(ids, 2))
    path = shortest(osm, a, b)
    if path is None or len(path) < 10:
      continue
    r = route_of(osm, path)
    if not 900 < r.length < 3000:
      continue
    slots = lane_slots.LaneSlots(r)
    targets = sum(1 for s in np.arange(0, r.length, 20.0) if slots.target(float(s), 10.0)[1] is not None)
    turns = len(r.maneuvers) - 2
    print(f"--start {a} --dest {b}: {r.length:.0f} m, {turns} turns, lane targets at {targets} of {int(r.length // 20)} points")


def nv12(path: str, w: int, h: int) -> bytes:
  from PIL import Image
  im = np.asarray(Image.open(path).convert("RGB").resize((w, h)), np.float32)
  r, g, b = im[..., 0], im[..., 1], im[..., 2]
  y = 0.257 * r + 0.504 * g + 0.098 * b + 16
  u = -0.148 * r - 0.291 * g + 0.439 * b + 128
  v = 0.439 * r - 0.368 * g - 0.071 * b + 128
  uv = np.stack([u[::2, ::2], v[::2, ::2]], axis=-1)
  return np.concatenate([y.clip(0, 255).astype(np.uint8).ravel(), uv.clip(0, 255).astype(np.uint8).ravel()]).tobytes()


class Openpilot:
  """The state the onroad view reads, engaged at a speed, with an alert when given."""
  SERVICES = ['deviceState', 'pandaStates', 'selfdriveState', 'carState', 'controlsState', 'modelV2', 'extrinsicsCalibration',
              'carParams', 'driverMonitoringState', 'deviceMotion']

  def __init__(self):
    self.pm = messaging.PubMaster(self.SERVICES)
    self.v = 12.0
    self.alert: tuple[str, str, str] | None = None  # text1, text2, size
    self.engaged = True
    self.yaw_rate = 0.0  # rad/s, clockwise positive

  def send(self):
    msgs = {s: messaging.new_message(s, valid=True) for s in self.SERVICES if s != 'pandaStates'}
    msgs['deviceState'].deviceState.started = True
    msgs['deviceState'].deviceState.deviceType = "tizi"
    ps = messaging.new_message('pandaStates', 1, valid=True)
    ps.pandaStates[0].ignitionLine = True
    ps.pandaStates[0].pandaType = log.PandaState.PandaType.tres
    msgs['pandaStates'] = ps
    ss = msgs['selfdriveState'].selfdriveState
    ss.enabled = ss.active = self.engaged
    ss.state = log.SelfdriveState.OpenpilotState.enabled if self.engaged else log.SelfdriveState.OpenpilotState.disabled
    if self.alert:
      ss.alertText1, ss.alertText2, ss.alertSize = self.alert
      ss.alertStatus = log.SelfdriveState.AlertStatus.normal
    cs = msgs['carState'].carState
    cs.vEgo, cs.vEgoCluster, cs.vCruiseCluster, cs.cruiseState.enabled = self.v, self.v, 56.0, True
    msgs['controlsState'].controlsState.deprecated.vCruise = 56.0
    cal = msgs['extrinsicsCalibration'].extrinsicsCalibration
    cal.calStatus, cal.rpyCalib = log.ExtrinsicsCalibration.Status.calibrated, [0.0, 0.0, 0.0]
    msgs['carParams'].carParams.openpilotLongitudinalControl = True
    md = msgs['modelV2'].modelV2
    x = (np.linspace(0, 1, 33) ** 2 * 120).tolist()
    md.position.x, md.position.y, md.position.z = x, [0.0] * 33, [0.0] * 33
    md.init('laneLines', 4)
    for line, y in zip(md.laneLines, (-5.4, -1.8, 1.8, 5.4), strict=True):
      line.x, line.y, line.z = x, [y] * 33, [0.0] * 33
    md.laneLineProbs = [0.5, 0.9, 0.9, 0.5]
    md.init('roadEdges', 2)
    for edge, y in zip(md.roadEdges, (-7.0, 7.0), strict=True):
      edge.x, edge.y, edge.z = x, [y] * 33, [0.0] * 33
    md.roadEdgeStds = [0.3, 0.3]
    md.acceleration.x = [0.0] * 33
    msgs['deviceMotion'].deviceMotion.angularVelocityDevice.z = self.yaw_rate  # device z points down
    for s, m in msgs.items():
      self.pm.send(s, m)


HEADING_SPAN = 6.0  # m either side over which the car's heading follows the route, cutting its corners as a car does


def car_heading(route: Route, s: float) -> float:
  """The car's heading s m along, clockwise from north."""
  a, b = point_at(route, s - HEADING_SPAN), point_at(route, s + HEADING_SPAN)
  return math.degrees(math.atan2(b[0] - a[0], b[1] - a[1])) % 360


def yaw_rate(route: Route, s: float, v: float) -> float:
  """rad/s, clockwise positive, at v m/s."""
  return math.radians(((car_heading(route, s + 0.5) - car_heading(route, s - 0.5) + 180) % 360 - 180) * v)


def place(route: Route, s: float, lane: int | None) -> np.ndarray:
  """Moves the car s m along the route, into our lane `lane` from the left (None: on the route's line); its position."""
  route.at = max(s - 5.0, 0.0)
  p = point_at(route, s)
  if lane is not None and route.lanes is not None:
    k = int(np.clip(np.searchsorted(route.along, s, side="right") - 1, 0, len(route.points) - 2))
    sec = route.lanes.opened_at(s, k)
    if sec is not None and sec.lanes:
      d = route.points[k + 1] - route.points[k]
      right = np.array([d[1], -d[0]]) / max(float(np.hypot(*d)), 1e-6)
      p = p + right * sec.offset(min(lane, sec.lanes - 1))
  route.locate(p)
  return p


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default=".")
  ap.add_argument("--find", action="store_true")
  ap.add_argument("--start", type=int)
  ap.add_argument("--dest", type=int)
  ap.add_argument("--metric", action="store_true")
  ap.add_argument("--fps", type=int, default=10)
  ap.add_argument("--disengaged", action="store_true", help="openpilot disengaged (the grey-blue border); names end _disengaged")
  ap.add_argument("--scenes", default="turn,lanes,arrive,idle",
                  help="any of turn, lanes, arrive, idle, at=<m>:<lane>, drive (through the first turn)")
  args = ap.parse_args()
  osm = load_map()
  if args.find:
    find(osm)
    return
  path = shortest(osm, args.start, args.dest)
  assert path is not None, "no route between those nodes"
  route = route_of(osm, path)
  slots = lane_slots.LaneSlots(route)
  print(f"route {route.length:.0f} m, maneuvers {[(m['type'], round(float(route.along[m['begin_shape_index']]))) for m in route.maneuvers]}")

  from openpilot.common.params import Params
  from openpilot.common.version import terms_version, training_version
  params = Params()
  params.put("HasAcceptedTerms", terms_version)
  params.put("CompletedTrainingVersion", training_version)
  params.put_bool("IsMetric", args.metric)

  import pyray as rl
  from msgq.visionipc import VisionIpcServer
  from openpilot.cereal.visionipc import VisionStreamType
  from openpilot.selfdrive.ui.layouts.main import MainLayout, MainState
  from openpilot.selfdrive.ui.ui_state import ui_state
  from openpilot.system.ui.lib.application import gui_app

  W, H = 1928, 1208
  vipc = VisionIpcServer("camerad")
  for stream in (VisionStreamType.VISION_STREAM_NARROW_ROAD, VisionStreamType.VISION_STREAM_WIDE_ROAD):
    vipc.create_buffers_with_sizes(stream, 4, W, H, W * H * 3 // 2, W, W * H)
  vipc.start_listener()
  frame = nv12(CAMERA, W, H)

  op = Openpilot()
  op.engaged = not args.disengaged
  nav = NavMessages()
  gui_app.init_window("nav ui demo", fps=args.fps)
  layout = MainLayout()
  ui_nav = layout._layouts[MainState.ONROAD].nav

  lanes_at = [float(s) for s in np.arange(0, route.length, 5.0) if slots.target(float(s), op.v)[1] is not None]
  turns = [float(route.along[m['begin_shape_index']]) for m in route.maneuvers if m['type'] in (9, 10, 11, 14, 15, 16)]

  def setup(name):
    """(s along to be at for the screenshot, lane, alert, route on) for a scene."""
    if name == "turn":
      return max(turns[0] - 180.0, 0.0), None, None, True
    if name == "lanes":
      s = lanes_at[len(lanes_at) // 3] if lanes_at else turns[-1] - 120.0
      here, target = slots.target(s, op.v)
      lane = None
      if target is not None and target.want:
        lane = min(target.want) - 1 if min(target.want) > 0 else max(target.want) + 1
      return s, lane, ("Steer Right to Start Lane Change Once Safe", "", log.SelfdriveState.AlertSize.small), True
    if name == "arrive":
      return max(turns[-1] + 10.0, route.length - 100.0), None, None, True
    if name == "idle":
      return 0.0, None, None, False
    if name == "drive":
      return turns[0] + DRIVE_PAST, None, None, True
    s, _, lane = name.removeprefix("at=").partition(":")
    return float(s), int(lane) if lane else None, None, True

  scenes = args.scenes.split(",")
  frame_id, k, cpu = 0, 0, []
  t0, lead = None, SETTLE
  drive_log: list[tuple[float, float, float, float]] = []  # t, the car's heading, the map's, the route segment's
  strip: list[str] = []
  for _, _, cpu_time in gui_app.render():
    if k >= len(scenes):
      break
    now = time.monotonic()
    target, lane, alert, on = setup(scenes[k])
    if t0 is None:
      t0 = now
      lead = (DRIVE_FROM + DRIVE_PAST) / DRIVE_SPEED if scenes[k] == "drive" else SETTLE
    op.v = DRIVE_SPEED if scenes[k] == "drive" else SPEED_SHOWN
    s = target - op.v * max(lead - (now - t0), 0.0)  # driving, to be there at the screenshot
    pos = place(route, s, lane)
    op.alert, op.yaw_rate = alert, yaw_rate(route, s, op.v)
    op.send()
    nav.update(route if on else None, op.v, osm, lambda r: slots, pose=(pos, car_heading(route, s)))
    vipc.send(VisionStreamType.VISION_STREAM_NARROW_ROAD, frame, frame_id, frame_id * 50_000_000, frame_id * 50_000_000)
    frame_id += 1
    ui_state.update()
    if now - t0 > 1.0:
      cpu.append(cpu_time)
    if scenes[k] == "drive" and ui_nav.pose.pos is not None:
      drive_log.append((now - t0, car_heading(route, s), ui_nav.pose.bearing, heading_at(route, route.at)))
      if len(drive_log) % max(int(args.fps * STRIP_EVERY), 1) == 0 and abs(s - turns[0]) < STRIP_NEAR:
        strip.append(save_frame(rl, gui_app, f"/tmp/navui_strip_{len(strip):02d}.png"))
    if now - t0 >= lead and not nav.busy and (ui_nav.roads or not on):  # the map drawn, roads and all
      name = scenes[k].replace('=', '_').replace(':', '_') + ("_disengaged" if args.disengaged else "")
      out = save_frame(rl, gui_app, os.path.join(args.out, f"navui_{name}.png"))
      print(f"saved {out} at {s:.0f} m; UI render {1000 * np.mean(cpu):.1f} ms mean, {1000 * np.max(cpu):.1f} max", flush=True)
      if scenes[k] == "drive":
        report_drive(drive_log, strip, os.path.join(args.out, "navui_drive_strip.jpg"))
      cpu.clear()
      k, t0 = k + 1, None
  gui_app.close()


DRIVE_FROM, DRIVE_PAST = 200.0, 40.0  # m before the first turn the drive starts, and past it it ends
DRIVE_SPEED = 8.0  # m/s
SPEED_SHOWN = 12.0  # m/s, in the other scenes
SETTLE = 4.0  # s of driving before a scene's screenshot
STRIP_EVERY, STRIP_NEAR = 1.0, 60.0  # s between the strip's frames, within this many m of the turn


def save_frame(rl, gui_app, out: str) -> str:
  rl.rl_draw_render_batch_active()
  img = rl.load_image_from_texture(gui_app._render_texture.texture)
  rl.image_flip_vertical(img)
  rl.export_image(img, out)
  rl.unload_image(img)
  return out


def report_drive(log_: list, frames: list[str], out: str):
  """How the map's heading followed the car's over the drive: the largest error, and the largest step from one frame to
  the next beyond the car's own (a snap); the route's segment headings, which the map used to follow, for comparison."""
  t, car, shown, seg = (np.array(c) for c in zip(*log_, strict=True))
  unwrap = lambda d: np.degrees(np.unwrap(np.radians(d)))  # noqa: E731
  err = (shown - car + 180) % 360 - 180
  snap = np.abs(np.diff(unwrap(shown)) - np.diff(unwrap(car)))
  seg_snap = np.abs(np.diff(unwrap(seg)) - np.diff(unwrap(car)))
  turned = unwrap(car)[-1] - unwrap(car)[0]
  worst = int(np.argmax(snap))
  print("; ".join([
    f"drive: {len(t)} frames over {t[-1]:.1f} s, car turned {turned:.0f} deg",
    f"map heading error max {np.abs(err).max():.1f} deg (rms {np.sqrt(np.mean(err ** 2)):.1f})",
    f"largest step beyond the car's {snap.max():.1f} deg (frame gap {1000 * (t[worst + 1] - t[worst]):.0f} ms)",
    f"route segment headings would step {seg_snap.max():.1f} deg",
  ]), flush=True)
  if frames:
    from PIL import Image
    crops = [Image.open(f).convert("RGB").crop((664, 222, 1066, 432)) for f in frames]
    sheet = Image.new("RGB", (sum(c.width for c in crops) + 6 * (len(crops) - 1), crops[0].height), (40, 40, 40))
    x = 0
    for c in crops:
      sheet.paste(c, (x, 0))
      x += c.width + 6
    sheet.thumbnail((2400, 400))
    sheet.save(out, quality=85)
    for f in frames:
      os.remove(f)
    print(f"saved {out} ({len(crops)} frames, {STRIP_EVERY} s apart)", flush=True)


if __name__ == "__main__":
  main()
