"""Blind-spot monitoring for the simulated car, as a Tesla's: whether a vehicle is beside the car in the lane to either
side, from the vehicles the plugin reports around it (its state's "nearby"). The flags reach openpilot as the Model 3's
DAS_status blind-spot signals (simulated_tesla.py), so openpilot's own lane change waits in preLaneChange, showing "Car
Detected in Blindspot", while that side is occupied; navd and the simulated driver wait on them too.

A side is occupied when a vehicle going our way overlaps its zone, by the vehicles' bodies (their model bounds turned to
their heading), not their middles:
- out from our side from `inner` m to the far edge of the lane beside (`outer` m where the lanes aren't known),
- along from `behind` m behind our rear bumper to `front` m ahead of our front bumper;
or is in that lane within `closing_behind` m behind our rear bumper, closing faster than `closing_speed` and due there
within `closing_time` s. A side's flag comes on at once and goes off once it has been clear for `clear_after` s, also
when its zone goes (the map's lanes beside flickering), so the flag never drops for a single reading.
Oncoming and crossing vehicles (heading over `heading` deg off ours), parked ones (stopped, nobody in the driver's
seat), and a side with no lane our way (the kerb, or across the centre line, where the car's lanes are known) never
count.

Defaults (GTA5_BLINDSPOT_TUNE, a JSON file of any of them, read again whenever it changes; GTA5_BLINDSPOT=0 turns the
monitor off):
- inner 1.0 m: where the zone starts out from our side, before the trim.
- outer: the lane beside's far edge, from the map's lanes along our route (GTA's lanes are 5.5 m, a 7 m reach on a
  2 m car); a car two lanes over is then a lane's width away. 4.5 m without them, as a US lane beside a centred car.
- lane_margin 0.3 m: on our route the zone is the lane beside itself (the map's besideLanes, its near and far edges),
  0.3 m in from each edge, so a car behind us off to one side of our own lane stays out whatever the lanes' width
  (GTA's freeway lanes run 5.5-6.6 m), while a car anywhere in the lane beside counts.
- trim 0.8 m: off the route, off both edges of the inner..outer span, keeping its centre: our own lane leaves ~1.75 m
  beside a centred car on a 5.5 m lane, which an untrimmed zone from 1.0 m would reach into.
- behind 8.0 m: about two car lengths; a car further back at our speed leaves a gap to merge into, and one closing
  on us is caught by the closing rule instead.
- front 0.0 m: as far as our front bumper rather than the mirrors, as a car alongside our bonnet is still one the lane
  change would move into; ahead of it the driving model sees for itself.
- closing_behind 30 m, closing_speed 3 m/s, closing_time 3 s: a lane change takes about 3 s to move over, so a car
  that would be alongside by then counts, while one barely gaining (a merge gap opening behind a slower car) doesn't.
- clear_after 0.5 s: holds the flag through a moment's gap between two cars, without delaying a real gap long.
- heading 60 deg: cars changing lanes or merging stay in; crossing traffic at junctions and oncoming cars don't.
- parked_speed 1.0 m/s: a vehicle slower than this with nobody driving it is parked; one waiting in traffic counts.
"""
import math
import os

import numpy as np

from openpilot.selfdrive.navd.planner import Tune

ENABLED = os.getenv("GTA5_BLINDSPOT", "1") != "0"
OWN_DIMS = (-1.0, 1.0, -2.4, 2.4)  # m: min x, max x, min y, max y of our car, until the plugin reports its model's
MAX_OUTER = 9.0  # m out from our side at most, from the map's lanes


class BlindSpotTune(Tune):
  """The monitor's settings (module docstring), from GTA5_BLINDSPOT_TUNE, read again whenever it changes."""
  DEFAULTS = {
    "inner": 1.0,  # m out from our side the zone starts
    "outer": 4.5,  # m out from our side it ends, where the lanes aren't known
    "trim": 0.8,  # m off both its inner and outer edges, keeping its centre, where the lane beside isn't known
    "lane_margin": 0.3,  # m in from both edges of the lane beside, where the map gives it
    "behind": 8.0,  # m behind our rear bumper it starts
    "front": 0.0,  # m ahead of our front bumper it ends (negative: behind it, as at the mirrors)
    "closing_behind": 30.0,  # m behind our rear bumper a vehicle closing on us counts from
    "closing_speed": 3.0,  # m/s faster than us
    "closing_time": 3.0,  # s to reach our rear bumper
    "clear_after": 0.5,  # s clear before the flag goes off
    "heading": 60.0,  # deg off ours, beyond which a vehicle isn't going our way
    "parked_speed": 1.0,  # m/s
  }


def footprint(v) -> tuple[float, float, float, float]:
  """A reported vehicle's body in our frame (x right, y forward of our origin): (x lo, x hi, y lo, y hi)."""
  x, y, heading, _, _, mnx, mxx, mny, mxy = v[:9]
  h = math.radians(heading)  # left of ours
  c, s = math.cos(h), math.sin(h)
  corners = ((mnx, mny), (mnx, mxy), (mxx, mny), (mxx, mxy))
  ex = [x + cx * c - cy * s for cx, cy in corners]
  ey = [y + cx * s + cy * c for cx, cy in corners]
  return min(ex), max(ex), min(ey), max(ey)


class BlindSpot:
  def __init__(self, tune: BlindSpotTune | None = None):
    self.tune = tune or BlindSpotTune(os.getenv("GTA5_BLINDSPOT_TUNE"))
    self.seen = {"left": -math.inf, "right": -math.inf}  # when each side was last occupied
    self.left = self.right = False
    self.zones: dict[str, tuple[float, float, float, float] | None] = {"left": None, "right": None}  # our frame, as footprint
    self.zone_lost = {"left": -math.inf, "right": -math.inf}  # when each side's zone last went
    self.reach = {"left": 0.0, "right": 0.0}  # m along (our frame) the rearmost vehicle that last set each side starts
    self.dims = OWN_DIMS
    self.last: dict | None = None  # the plugin's "nearby" last looked at

  def update(self, state: dict, now: float) -> tuple[bool, bool]:
    """The flags (left, right) from the plugin's state, after the bridge's _map_route (its "lane" and "beside")."""
    if self.tune.refresh(now):
      print(f"gta5: blind spot tune {self.tune.changed()}")
    nearby = state.get("nearby") if ENABLED else None
    if not nearby:
      self.left = self.right = False
      self.zones = {"left": None, "right": None}
      self.last = None
      return False, False
    if nearby is not self.last:  # a new game state: the bridge steps faster than the game sends them
      self.last = nearby
      self._look(state, nearby, now)
    t = self.tune
    self.left, self.right = (now - self.seen[side] < t.clear_after and (self.zones[side] is not None or now - self.zone_lost[side] < t.clear_after)
                             for side in ("left", "right"))
    return self.left, self.right

  def _look(self, state: dict, nearby: dict, now: float):
    """Each side's zone, and when it was last occupied, from a game state's vehicles."""
    t = self.tune
    self.dims = tuple(nearby.get("dims") or OWN_DIMS)
    mnx, mxx, mny, mxy = self.dims
    v_ego = state.get("vEgo", 0.0)
    outer = self._outer(state, (mxx - mnx) / 2)
    along = (mny - t.behind, mxy + t.front)
    for side, sign, edge in (("left", -1.0, mnx), ("right", 1.0, mxx)):
      if outer[side] is None:
        if self.zones[side] is not None:
          self.zone_lost[side] = now
        self.zones[side] = None
        continue
      lane = (state.get("besideLanes") or [None, None])[0 if side == "left" else 1]
      if lane is not None:  # the lane beside itself, from the map: whatever its width, our own lane stays out
        near, far = lane[0] + t.lane_margin, max(min(lane[1], MAX_OUTER + (mxx - mnx) / 2) - t.lane_margin, lane[0] + t.lane_margin + 1.0)
        lo, hi = sorted((sign * near, sign * far))
      else:
        lo, hi = sorted((edge + sign * t.inner, edge + sign * max(outer[side], t.inner + 1.0)))
        trim = min(t.trim, max((hi - lo - 1.0) / 2, 0.0))  # narrowed about its centre, keeping it at least 1 m wide
        lo, hi = lo + trim, hi - trim
      self.zones[side] = (lo, hi, *along)
      hits = [v for v in nearby.get("v") or [] if self._occupies(v, lo, hi, along, mny, v_ego)]
      if hits:
        self.seen[side] = now
        # how far back the vehicles that set it reach, for the overlay: a closing one is behind the zone
        self.reach[side] = min(footprint(v)[2] for v in hits)

  def _outer(self, state: dict, half_width: float) -> dict[str, float | None]:
    """m out from our side each zone reaches, None for a side with no lane our way: by the map's lanes along our route,
    else by its lane match off it (not the plugin's own lane reading, which counts one of GTA's links side by side)."""
    beside, matched = state.get("beside"), state.get("laneMap") or {}
    if beside is not None:
      return {side: None if d is None else min(d - half_width, MAX_OUTER) for side, d in zip(("left", "right"), beside, strict=True)}
    out: dict[str, float | None] = {"left": self.tune.outer, "right": self.tune.outer}
    if matched.get("kind") == "own" and matched.get("lanes") and matched.get("lane") is not None:
      if matched["lane"] <= 0:
        out["left"] = None
      if matched["lane"] >= matched["lanes"] - 1:
        out["right"] = None
    return out

  def _occupies(self, v, lo: float, hi: float, along: tuple[float, float], rear: float, v_ego: float) -> bool:
    t = self.tune
    if abs(v[2]) > t.heading:
      return False  # oncoming, or crossing
    if math.hypot(v[3], v[4]) < t.parked_speed and not v[9]:
      return False  # parked
    x0, x1, y0, y1 = footprint(v)
    if x1 < lo or x0 > hi:
      return False
    if y1 >= along[0] and y0 <= along[1]:
      return True
    gap, closing = rear - y1, v[4] - v_ego
    return 0.0 < gap <= t.closing_behind and closing > t.closing_speed and gap / closing < t.closing_time

  def overlay(self, state: dict, lead: float = 0.0) -> list[tuple[str, np.ndarray]]:
    """Each side's zone for the map debug overlay, in the world (x, y, the road's z): z its outline, Z filled while
    occupied; from where the car will be `lead` s on."""
    if "pos" not in state:
      return []
    h = math.radians(state.get("heading", 0.0))
    right, fwd = np.array([math.cos(h), math.sin(h)]), np.array([-math.sin(h), math.cos(h)])
    pos = np.array(state["pos"][:2], dtype=float) + fwd * max(state.get("vEgo", 0.0), 0.0) * lead
    z = float(state["pos"][2]) - 0.6  # the road under the car's origin (paths.CAR_HEIGHT)
    out = []
    for side, on in (("left", self.left), ("right", self.right)):
      zone = self.zones[side]
      if zone is None:
        continue
      x0, x1, y0, y1 = zone

      def world(pts):
        xy = pos + np.outer([p[0] for p in pts], right) + np.outer([p[1] for p in pts], fwd)
        return np.column_stack([xy, np.full(len(xy), z)])

      out.append(("z", world([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)])))
      if on:
        # back to the vehicle that set it, so a car closing from behind the zone shows where it is
        out.append(("Z", world([((x0 + x1) / 2, min(y0, self.reach[side])), ((x0 + x1) / 2, y1)])))
    return out
