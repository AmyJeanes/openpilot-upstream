#!/usr/bin/env python3
import os
import time

import numpy as np
import requests

import openpilot.cereal.messaging as messaging
from openpilot.common.api import API_HOST, Api
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.navigation import lane_guidance
from openpilot.selfdrive.navigation.helpers import latest_gps_source
from openpilot.selfdrive.navigation.route import along_path, local_meters

REROUTE_DISTANCE = 25
MANEUVER_TRANSITION_THRESHOLD = 10
REROUTE_COUNTER_MIN = 3
ROUTE_RESEND_PERIOD = 10  # updates (seconds); a restarted mapsd gets the route within this
RECOMPUTE_HOLD = 10  # updates (seconds) between route requests
GUIDANCE_RATE = 10  # Hz the lane guidance runs at (the route's updates stay at 1 Hz)
LANE_CHANGING = ("laneChangeStarting", "laneChangeFinishing")


def set_banner(instruction, banners, distance_to_maneuver):
  """Fill the banner fields from the banner to show now: the last one still ahead of the ego, else the first."""
  if not banners:
    return
  banner = next((b for b in reversed(banners) if distance_to_maneuver < b['distanceAlongGeometry']), banners[0])
  instruction.showFull = distance_to_maneuver < banner['distanceAlongGeometry']
  primary = banner['primary']
  instruction.maneuverPrimaryText = primary.get('text') or ''
  instruction.maneuverType = primary.get('type') or ''
  instruction.maneuverModifier = primary.get('modifier') or ''
  if banner.get('secondary') is not None:
    instruction.maneuverSecondaryText = banner['secondary']['text']


class RouteEngine:
  def __init__(self, sm, pm, params=None):
    self.sm = sm
    self.pm = pm
    self.params = params if params is not None else Params()

    self.last_position = None  # (longitude, latitude)
    self.last_bearing = None

    self.nav_destination = None
    self.step_idx = None
    self.route = None
    self.route_geometry = None  # an (n, 2) array of (longitude, latitude) per step

    self.recompute_hold = 0
    self.reroute_counter = 0

    self.mapbox_token = os.environ.get("MAPBOX_TOKEN")
    # a Mapbox-compatible host instead (MAPBOX_HOST, else the NavMapboxHost param): a self-hosted router, or a
    # simulator's (GTA's tools/sim/bridge/gta5/mapbox_server.py); it gets MAPBOX_TOKEN if set, else no token
    self.custom_host = (os.environ.get("MAPBOX_HOST") or self.params.get("NavMapboxHost") or "").rstrip('/') or None
    self.api = Api(self.params.get("DongleId")) if self.mapbox_token is None and self.custom_host is None else None
    self.mapbox_host = self.custom_host or ("https://api.mapbox.com" if self.mapbox_token is not None else API_HOST.rstrip('/') + '/v1/nav')

    # step 1 lane guidance (lane_guidance.py), behind the NavLaneGuidance param or NAVD_LANE_GUIDANCE=1
    self.guidance = lane_guidance.LaneGuidance(lane_guidance.tune_from_env()) if lane_guidance.enabled(self.params) else None
    self.guided: lane_guidance.GuidanceOutputs | None = None
    self.route_steps_along: list[float] = []  # m along the route to each step's start
    self.updates = 0  # full (1 Hz) updates, while the guidance runs at GUIDANCE_RATE

  def update(self, full: bool = True):
    """A step: the route (requested, followed, resent: `full`, at 1 Hz) and the instruction; with the lane guidance,
    run at GUIDANCE_RATE, its step and the instruction every time."""
    self.sm.update(0)

    source = latest_gps_source(self.sm)
    if source is not None:
      gps = self.sm[source]
      self.last_position, self.last_bearing = (gps.longitude, gps.latitude), gps.bearingDeg
    route = self.route
    try:
      if full:
        self.recompute_route(gps_ok=source is not None)
      self.guide()
      self.send_instruction()
    except Exception:
      cloudlog.exception("navd.failed_to_compute")
    if not full:
      return
    # mapsd dedupes repeats, so a periodic resend is how a restarted mapsd gets the route
    tick = self.sm.frame if self.guidance is None else self.updates
    self.updates += 1
    if self.route is not route or tick % ROUTE_RESEND_PERIOD == 0:
      self.send_route()

  def recompute_route(self, gps_ok: bool):
    if self.last_position is None:
      return

    destination = self.params.get("NavDestination")
    if destination is None:
      self.clear_route()
      self.recompute_hold = 0
      return
    new_destination = (destination['longitude'], destination['latitude'])

    # Don't recompute when GPS drifts in tunnels
    if not gps_ok and self.step_idx is not None:
      return

    self.recompute_hold -= 1
    if (self.should_recompute() or new_destination != self.nav_destination) and self.recompute_hold <= 0:
      self.recompute_hold = RECOMPUTE_HOLD
      self.calculate_route(new_destination)
      self.reroute_counter = 0

  def calculate_route(self, destination):
    assert self.last_position is not None
    cloudlog.warning(f"Calculating route {self.last_position} -> {destination}")
    self.nav_destination = destination

    lang = self.params.get('LanguageSetting')
    if lang is not None:
      lang = lang.replace('main_', '')

    token = self.mapbox_token
    if token is None and self.api is not None:
      token = self.api.get_token()

    params = {
      **({'access_token': token} if token is not None else {}),
      'geometries': 'geojson',
      'overview': 'full',
      'steps': 'true',
      'banner_instructions': 'true',
      'alternatives': 'false',
      'language': lang,
      **({'annotations': 'maxspeed,congestion'} if self.guidance is not None else {}),
    }
    if self.last_bearing is not None:
      params['bearings'] = f"{(self.last_bearing + 360) % 360:.0f},90;"

    coords_str = ';'.join(f'{lon},{lat}' for lon, lat in (self.last_position, destination))
    url = self.mapbox_host + '/directions/v5/mapbox/driving-traffic/' + coords_str
    try:
      resp = requests.get(url, params=params, timeout=10)
      if resp.status_code != 200:
        cloudlog.event("API request failed", status_code=resp.status_code, text=resp.text, error=True)
      resp.raise_for_status()

      r = resp.json()
      if len(r['routes']):
        self.route = r['routes'][0]['legs'][0]['steps']
        self.route_geometry = [np.array(step['geometry']['coordinates'], dtype=np.float64).reshape((-1, 2)) for step in self.route]
        self.step_idx = 0
        self.route_steps_along = np.r_[0.0, np.cumsum([step['distance'] for step in self.route])].tolist()
        if self.guidance is not None:
          self.guidance.set_route(r['routes'][0])
      else:
        cloudlog.warning("Got empty route response")
        self.clear_route()

    except requests.exceptions.RequestException:
      cloudlog.exception("failed to get route")
      self.clear_route()

  def send_instruction(self):
    msg = messaging.new_message('navInstruction', valid=True)

    if self.step_idx is None:
      msg.valid = False
      self.pm.send('navInstruction', msg)
      return

    assert self.route is not None and self.route_geometry is not None and self.last_position is not None
    step = self.route[self.step_idx]
    along_geometry, _ = along_path(self.route_geometry[self.step_idx], self.last_position)
    distance_to_maneuver_along_geometry = step['distance'] - along_geometry

    # Banner instructions are for the following maneuver step, don't use empty last step
    banner_step = step
    if not len(banner_step['bannerInstructions']) and self.step_idx == len(self.route) - 1:
      banner_step = self.route[max(self.step_idx - 1, 0)]

    # Current instruction
    msg.navInstruction.maneuverDistance = distance_to_maneuver_along_geometry
    self.fill_guidance(msg.navInstruction)
    set_banner(msg.navInstruction, banner_step['bannerInstructions'], distance_to_maneuver_along_geometry)

    # Total remaining distance and time: the rest of this step, then every step after it
    remaining = 1.0 - along_geometry / max(step['distance'], 1)
    future = self.route[self.step_idx + 1:]
    msg.navInstruction.distanceRemaining = step['distance'] * remaining + sum(s['distance'] for s in future)
    msg.navInstruction.timeRemaining = step['duration'] * remaining + sum(s['duration'] for s in future)

    self.pm.send('navInstruction', msg)

    # Transition to next route segment
    if distance_to_maneuver_along_geometry < -MANEUVER_TRANSITION_THRESHOLD:
      if self.step_idx + 1 < len(self.route):
        self.step_idx += 1
        self.recompute_hold = 0
      else:
        cloudlog.warning("Destination reached")

        # Clear route if driving away from destination
        if np.linalg.norm(local_meters(self.last_position, self.nav_destination)) > REROUTE_DISTANCE:
          self.params.remove("NavDestination")
          self.clear_route()

  def send_route(self):
    coords = np.concatenate(self.route_geometry) if self.route_geometry is not None else np.empty((0, 2))
    msg = messaging.new_message('navRoute', valid=True)
    msg.navRoute.coordinates = [{'longitude': lon, 'latitude': lat} for lon, lat in coords.tolist()]
    self.pm.send('navRoute', msg)

  def clear_route(self):
    self.route = None
    self.route_geometry = None
    self.step_idx = None
    self.nav_destination = None
    if self.guidance is not None and self.guidance.route is not None:
      self.guidance.set_route(None)
      self.guided = None
      self.put_desire("")

  # *** step 1 lane guidance ***

  def progress(self) -> float | None:
    """m along the whole route to the car (the current step's start, then along its geometry)."""
    if self.step_idx is None or self.route_geometry is None or self.last_position is None:
      return None
    along, _ = along_path(self.route_geometry[self.step_idx], self.last_position)
    return self.route_steps_along[self.step_idx] + min(along, self.route[self.step_idx]['distance'])

  def guide(self):
    """The lane guidance's step: from the route, carState and the driving model's lane head; NavDesire written as it
    changes."""
    if self.guidance is None:
      return
    along = self.progress()
    if along is None or self.guidance.route is None:
      self.guided = None
      return
    cs, md = self.sm['carState'], self.sm['modelV2']
    head = md.laneHead
    fresh = self.sm.alive['modelV2'] and self.sm.valid['modelV2'] and head.laneCount > 0 and head.laneIdx >= 0
    blinker = "left" if cs.leftBlinker and not cs.rightBlinker else "right" if cs.rightBlinker and not cs.leftBlinker else None
    stage = self.params.get("NavLaneChangeStage")
    inp = lane_guidance.GuidanceInputs(time.monotonic(), along, cs.vEgo, [head.laneIdx, head.laneCount, head.prob] if fresh else None,
                                       blinker, str(md.meta.laneChangeState) in LANE_CHANGING, cs.leftBlindspot,
                                       cs.rightBlindspot, int(stage) if stage is not None else 1)
    self.guided = self.guidance.update(inp)
    for desire in self.guided.desires:
      self.put_desire(desire)
    for event in self.guided.events:
      cloudlog.event("navd.request", **event)

  def put_desire(self, desire: str):
    if desire:
      self.params.put("NavDesire", desire)
    else:
      self.params.remove("NavDesire")

  def fill_guidance(self, instruction):
    """navInstruction's request, lanes and laneDistance from the lane guidance's last step."""
    g = self.guided
    if g is None:
      return
    if g.request is not None:
      r = g.request
      req = instruction.request
      req.kind, req.side, req.reason = r["kind"], r["side"], r["reason"]
      req.distance, req.laneChanges, req.stage, req.id, req.approved = r["distance"], r["laneChanges"], r["stage"], r["id"], r["approved"]
    if g.lanes is not None:
      lanes = instruction.init("lanes", len(g.lanes))
      for lane, want in zip(lanes, g.lanes, strict=True):
        lane.directions = want["directions"]
        lane.active, lane.activeDirection, lane.current = want["active"], want["activeDirection"], want["current"]
    instruction.laneDistance = g.lane_distance

  def should_recompute(self):
    if self.step_idx is None:
      return True

    assert self.route is not None and self.route_geometry is not None and self.last_position is not None
    # Don't recompute in last segment, assume destination is reached
    if self.step_idx == len(self.route) - 1:
      return False

    # TODO: Check for going wrong way in segment
    if along_path(self.route_geometry[self.step_idx], self.last_position)[1] > REROUTE_DISTANCE:
      self.reroute_counter += 1
    else:
      self.reroute_counter = 0
    return self.reroute_counter > REROUTE_COUNTER_MIN


def main():
  pm = messaging.PubMaster(['navInstruction', 'navRoute'])
  params = Params()
  guided = lane_guidance.enabled(params)
  sm = messaging.SubMaster(['gpsLocation', 'gpsLocationExternal'] + (['carState', 'modelV2'] if guided else []),
                           ignore_alive=['carState', 'modelV2'], ignore_valid=['carState', 'modelV2'])

  rate = GUIDANCE_RATE if guided else 1
  rk = Ratekeeper(float(rate))
  route_engine = RouteEngine(sm, pm, params)
  while True:
    route_engine.update(full=rk.frame % rate == 0)
    rk.keep_time()


if __name__ == "__main__":
  main()
