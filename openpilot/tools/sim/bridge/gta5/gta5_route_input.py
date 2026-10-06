"""The route input: a fixed-size vector telling a route-conditioned driving model (gta5-train --route) where the route
goes. The bridge feeds it to modeld from its live route (gta5_world.py, through selfdrive/modeld/route_input.py), and
gta5-train labels recorded segments with the same encoder, so it depends only on a router.Route and the car's place on it.

Layout (ROUTE_LEN = 173 floats; all zero = no route, which is also what a dropped input looks like):
- [0:150] NAV: comma's 2023 `nav_instructions` (openpilot 75a69e12b^ modeld.py): multi-hot, index bin * 3 + direction,
  50 bins of 20 m from 500 m behind to 500 m ahead (bin 25 = 0..20 m ahead), direction 0 other / 1 left / 2 right.
  Every Valhalla maneuver of the route lights its bin (LEFT / RIGHT types, the rest "other", start and destination
  included). Bins are floor(d / 20) + 25, where comma's int() truncation made bin 25 span -20..+20 m.
- [150] PRESENT: 1 whenever a route is given.
- [151:161] HEADING: the route's direction at 10, 20, ..., 100 m ahead (the chord over +-10 m) relative to the car's
  heading, right positive, in units of 90 deg (clipped to +-2). Beyond the route's end its last direction is held.
- [161:171] NEXT: the next maneuver nav acts on (gta5_expert.maneuvers: turns and keeps), kept until 20 m past it
  (or until the next is within 60 m), within 500 m:
  161 present; 162 side (-1 left, +1 right); 163 heading change, right positive, / 90 deg;
  164 m to it / 100 (clipped -0.5..3); 165 m to its junction entry / 100 (gta5_nav.junction_entry: the stop line, else
  the first junction node; the maneuver itself for keeps; clipped -0.5..3);
  166 / 167 lanes before / after it (Route.lanes_at) / 6, clipped to 1; 168 / 169 the target lanes nav aims for
  (gta5_nav Turn.lanes / Fork.lanes): the centres of the first and last as a fraction of the road from the left,
  (lane + 0.5) / lanes; 170 lane data present (lanes before > 0).
- [171:173] STOP: 171 present; 172 m to the next stop line on the route (Route.stops, from 2 m behind, within 300 m)
  / 100. With router.STOP_DIRECTION off they include the far side of junctions: label with the bridge's setting.
"""
import math
from typing import NamedTuple

import numpy as np

from openpilot.tools.sim.bridge.gta5 import gta5_expert, gta5_nav

NAV_BINS, BIN_M, BIN_ZERO = 50, 20.0, 25
NAV_LEN = NAV_BINS * 3
PRESENT = NAV_LEN
HEADING = slice(151, 161)
NEXT = slice(161, 171)
STOP = slice(171, 173)
ROUTE_LEN = 173
HEADING_AHEAD = np.arange(10.0, 101.0, 10.0)
# headings sampled from 30 m behind to 130 m ahead, so gta5-train's +-1 bin (20 m) distance jitter can shift them consistently
HEADING_EXT_AHEAD = np.arange(-30.0, 131.0, 10.0)
HEADING_EXT_NOMINAL = slice(4, 14)
HEADING_UNIT = 90.0  # deg
DIST_UNIT = 100.0  # m
NEXT_PAST = 20.0  # m past a maneuver it stays the next one
NEXT_AHEAD = 500.0
NEXT_SOON = 60.0  # m: a maneuver this near takes over from one just passed
STOP_BEHIND, STOP_AHEAD = 2.0, 300.0
TANGENT_M = 10.0  # GTA routes jog sideways between lanes and road nodes: a shorter chord reads those as turns
LANES_UNIT = 6.0

# Valhalla maneuver types; U-turns count as other, as comma's "uturn" modifier did
LEFT = {14, 15, 16, 19, 21, 24}  # sharp left, left, slight left, ramp left, exit left, stay (keep) left
RIGHT = {9, 10, 11, 18, 20, 23}


def direction(kind: int | None) -> int:
  return 1 if kind in LEFT else 2 if kind in RIGHT else 0


def wrap(deg):
  return (np.asarray(deg) + 180.0) % 360.0 - 180.0


class RouteInput:
  """Encodes one route (router.Route, with GTA's road data for lanes, stops and junctions). Everything about its
  maneuvers is worked out once here; `encode` per frame is cheap."""

  def __init__(self, route):
    self.points = np.asarray(route.points, float)
    self.along = np.asarray(route.along, float)
    self.length = float(self.along[-1])
    # NAV: (m along, direction) for every Valhalla maneuver
    self.nav = np.array([(self.along[m["begin_shape_index"]], direction(m.get("type"))) for m in route.maneuvers
                         if 0 <= m.get("begin_shape_index", -1) < len(self.along)], float).reshape(-1, 2)
    self.stops = np.sort(np.asarray(route.stops, float))
    # NEXT: per maneuver nav acts on, [along, side, change, entry, lanes in, lanes out, target lo, target hi]
    rows = []
    for m in gta5_expert.maneuvers(route):
      s = m.along
      before, here = (self.point(v) for v in (s - gta5_expert.HEADING_SPAN, s))
      change = -float(wrap(m.exit_heading - gta5_expert.heading_between(before, here)))  # right positive
      side = -1.0 if m.desire.endswith("Left") else 1.0
      n_in, n_out = self._lanes_at(route, s, False), self._lanes_at(route, s, True)
      if m.turn:
        entry = gta5_nav.junction_entry(s, list(route.stops), list(route.junctions))[0]
        lo, hi = gta5_nav.Turn(s, "left" if side < 0 else "right", m.exit_heading).lanes(max(n_in, 1))
      else:
        entry = s
        forks = [f for f in route.forks if abs(f.along - s) < 15.0]
        if forks:
          f = min(forks, key=lambda f: abs(f.along - s))
          lo, hi = gta5_nav.Fork(0.0, f.side, f.lanes, f.lanes_in, f.keep, f.other, f.slip).lanes(max(n_in, 1))
        else:
          lo, hi = 0, max(n_in, 1) - 1
      rows.append([s, side, change, entry, n_in, n_out, lo, hi])
    self.next = np.array(rows, float).reshape(-1, 8)

  @staticmethod
  def _lanes_at(route, s: float, after: bool) -> int:
    at = route.at
    route.at = 0.0
    try:
      return int(route.lanes_at(s, after))
    finally:
      route.at = at

  def point(self, s):
    s = np.asarray(s, float)
    return np.stack([np.interp(s, self.along, self.points[:, 0]), np.interp(s, self.along, self.points[:, 1])], axis=-1)

  def heading_at(self, s) -> np.ndarray:
    """The route's game heading (deg counterclockwise from north) at s m along, over +-TANGENT_M; held past the ends."""
    s = np.asarray(s, float)
    a = np.clip(s - TANGENT_M, 0.0, max(self.length - 2 * TANGENT_M, 0.0))
    b = np.minimum(a + 2 * TANGENT_M, self.length)
    d = self.point(b) - self.point(a)
    return np.degrees(np.arctan2(-d[..., 0], d[..., 1]))

  def off(self, s: float, pos: np.ndarray) -> float:
    return float(np.hypot(*(np.asarray(pos, float)[:2] - self.point(s))))

  def headings_ext(self, s: float, heading: float) -> np.ndarray:
    """[17] the HEADING feature over HEADING_EXT_AHEAD (the encoded part is HEADING_EXT_NOMINAL)."""
    rel = -wrap(self.heading_at(s + HEADING_EXT_AHEAD) - heading)  # right positive
    return np.clip(rel / HEADING_UNIT, -2.0, 2.0).astype(np.float32)

  def encode(self, s: float, heading: float) -> np.ndarray:
    """[ROUTE_LEN] for the car s m along the route, heading `heading` (game degrees, as the plugin's)."""
    out = np.zeros(ROUTE_LEN, np.float32)
    out[PRESENT] = 1.0
    if len(self.nav):
      d = self.nav[:, 0] - s
      b = BIN_ZERO + np.floor(d / BIN_M).astype(int)
      ok = (b >= 0) & (b < NAV_BINS)
      out[b[ok] * 3 + self.nav[ok, 1].astype(int)] = 1.0
    out[HEADING] = self.headings_ext(s, heading)[HEADING_EXT_NOMINAL]
    if len(self.next):
      d = self.next[:, 0] - s
      ahead = np.flatnonzero((d > -NEXT_PAST) & (d < NEXT_AHEAD))
      if len(ahead) > 1 and d[ahead[0]] < 0 and d[ahead[1]] < NEXT_SOON:
        ahead = ahead[1:]  # one just passed gives way to one coming up
      if len(ahead):
        a, side, change, entry, n_in, n_out, lo, hi = self.next[ahead[0]]
        n = max(n_in, 1.0)
        out[NEXT] = [1.0, side, np.clip(change / HEADING_UNIT, -2, 2), np.clip((a - s) / DIST_UNIT, -0.5, 3.0),
                     np.clip((entry - s) / DIST_UNIT, -0.5, 3.0), min(n_in / LANES_UNIT, 1.0), min(n_out / LANES_UNIT, 1.0),
                     (lo + 0.5) / n, (hi + 0.5) / n, float(n_in > 0)]
    if len(self.stops):
      d = self.stops - s
      d = d[(d > -STOP_BEHIND) & (d < STOP_AHEAD)]
      if len(d):
        out[STOP] = [1.0, d.min() / DIST_UNIT]
    return out


class Next(NamedTuple):
  side: str  # L or R
  change: float  # deg, right positive
  dist: float  # m
  entry: float  # m
  lanes_in: int
  lanes_out: int
  target: tuple[float, float]  # first and last target lane centres, as a fraction of the road from the left
  lane_data: bool


class Decoded(NamedTuple):
  nav: np.ndarray  # [NAV_BINS, 3] bool: other, left, right
  present: bool
  heading: np.ndarray  # deg, right positive, at HEADING_AHEAD
  next: Next | None
  stop: float | None  # m


def decode(vec: np.ndarray) -> Decoded:
  """The route input back in plain units."""
  vec = np.asarray(vec, np.float32)
  nx = vec[NEXT]
  nxt = Next("L" if nx[1] < 0 else "R", float(nx[2] * HEADING_UNIT), float(nx[3] * DIST_UNIT), float(nx[4] * DIST_UNIT),
             round(float(nx[5] * LANES_UNIT)), round(float(nx[6] * LANES_UNIT)), (float(nx[7]), float(nx[8])), bool(nx[9])) if nx[0] else None
  stop = float(vec[STOP.start + 1] * DIST_UNIT) if vec[STOP.start] else None
  return Decoded(vec[:NAV_LEN].reshape(NAV_BINS, 3) > 0, bool(vec[PRESENT]), vec[HEADING] * HEADING_UNIT, nxt, stop)


def describe(vec: np.ndarray, lo_bin: int = 20, hi_bin: int = 50) -> str:
  """One line: NAV bins lo_bin..hi_bin as characters (. none, o other, L, R; several: *), then NEXT and STOP."""
  d = decode(vec)
  chars = []
  for b in range(lo_bin, hi_bin):
    lit = np.flatnonzero(d.nav[b])
    chars.append("." if not len(lit) else "*" if len(lit) > 1 else "oLR"[lit[0]])
  nx = d.next
  nxt = (f"next {nx.side} {nx.change:+4.0f}deg at {nx.dist:5.0f} m entry {nx.entry:5.0f} m " +
         f"lanes {nx.lanes_in}->{nx.lanes_out} tgt {nx.target[0]:.2f}-{nx.target[1]:.2f}") if nx else "next -"
  stop = f"stop {d.stop:4.0f} m" if d.stop is not None else "stop -"
  hd = " ".join(f"{v:+4.0f}" for v in d.heading[::3])
  return f"{''.join(chars)} | hdg {hd} | {nxt} | {stop}"


assert NEXT.stop == STOP.start and STOP.stop == ROUTE_LEN and HEADING.start == PRESENT + 1 and NEXT.start == HEADING.stop
assert math.isclose(HEADING_EXT_AHEAD[HEADING_EXT_NOMINAL][0], HEADING_AHEAD[0])
