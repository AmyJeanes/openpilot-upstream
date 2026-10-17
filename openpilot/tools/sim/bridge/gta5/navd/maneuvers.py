"""The route's maneuvers the car must do something for (turns and keeps), from its Valhalla maneuvers and shape: the
route input's NEXT and lane slots, and the expert driver's indicators, all read them alike."""
import math

import numpy as np

HEADING_SPAN = 20.0  # m before and after a maneuver for its headings
TURN_ANGLE = 45.0  # deg: a turn; less is a keep
BEND_ANGLE = 15.0  # deg: less than this is straight on
# A turn onto an unnamed way followed this soon by another turn is one move through a junction, by its links across it
# (a divided road's median, a jog between offset roads), which the router doesn't fold into one maneuver
JUNCTION_LINK = 25.0  # m
# Valhalla maneuver types
TURNS = {9: "Right", 10: "Right", 11: "Right", 14: "Left", 15: "Left", 16: "Left"}
FORKS = {18: "Right", 19: "Left", 20: "Right", 21: "Left", 23: "Right", 24: "Left"}
SIGNALED_FORKS = {18, 19, 20, 21}  # ramps and exits
CONTINUE = 8
# a merged turn's type by the deg it turns (Valhalla's own bounds): continue, slight, turn, sharp, U-turn
RIGHT_TYPES = ((10.0, CONTINUE), (44.0, 9), (135.0, 10), (159.0, 11), (180.0, 12))
LEFT_TYPES = ((10.0, CONTINUE), (44.0, 16), (135.0, 15), (159.0, 14), (180.0, 13))


def wrap(deg: float) -> float:
  return (deg + 180) % 360 - 180


class Maneuver:
  """A maneuver on the route the car must do something for: its place (m along), desire and whether it's signaled."""
  def __init__(self, along: float, desire: str, signal: bool, exit_heading: float, turn: bool):
    self.along = along
    self.desire = desire
    self.signal = signal
    self.exit_heading = exit_heading
    self.turn = turn


def route_point(route, s: float) -> np.ndarray:
  return np.array([np.interp(s, route.along, route.points[:, 0]), np.interp(s, route.along, route.points[:, 1])])


def heading_between(a: np.ndarray, b: np.ndarray) -> float:
  """Degrees counterclockwise from north (the map's x east, y north)."""
  return math.degrees(math.atan2(-(b[0] - a[0]), b[1] - a[1]))


def junction_maneuvers(route) -> list[dict]:
  """The route's Valhalla maneuvers with each run of turns through a junction's links (a turn onto an unnamed way, then
  another within JUNCTION_LINK m) made one: at the first's place, of the type the whole run turns (a jog that comes
  back straight continues), onto the last's road. It keeps the last's shape index as `exit_shape_index`, where the
  move's way out is."""
  mans, n = route.maneuvers, len(route.along)

  def along(m):
    return float(route.along[min(m.get("begin_shape_index", 0), n - 1)])

  out, k = [], 0
  while k < len(mans):
    run = [mans[k]]
    while (run[-1].get("type") in TURNS and not run[-1].get("street_names") and k + len(run) < len(mans)
           and mans[k + len(run)].get("type") in TURNS and along(mans[k + len(run)]) - along(run[0]) <= JUNCTION_LINK):
      run.append(mans[k + len(run)])
    k += len(run)
    if len(run) == 1:
      out.append(run[0])
      continue
    first, last = run[0], run[-1]
    s0, s1 = along(first), along(last)
    turned = wrap(heading_between(route_point(route, s1), route_point(route, s1 + HEADING_SPAN)) -
                  heading_between(route_point(route, s0 - HEADING_SPAN), route_point(route, s0)))
    kind = next(t for limit, t in (LEFT_TYPES if turned > 0 else RIGHT_TYPES) if abs(turned) <= limit)
    out.append({**last, "type": kind, "begin_shape_index": first.get("begin_shape_index", 0),
                "exit_shape_index": last.get("begin_shape_index", 0), "length": sum(m.get("length", 0.0) for m in run),
                **({"time": sum(m["time"] for m in run)} if all("time" in m for m in run) else {})})
  return out


def maneuvers(route) -> list[Maneuver]:
  out = []
  for m in junction_maneuvers(route):
    kind, i = m.get("type"), m.get("begin_shape_index", 0)
    if kind not in TURNS and kind not in FORKS or i >= len(route.along):
      continue
    s = float(route.along[i])
    out_s = float(route.along[min(m.get("exit_shape_index", i), len(route.along) - 1)])  # past a junction's links
    before, here = (route_point(route, v) for v in (s - HEADING_SPAN, s))
    exit_heading = heading_between(route_point(route, out_s), route_point(route, out_s + HEADING_SPAN))
    change = wrap(exit_heading - heading_between(before, here))
    if kind in TURNS:
      if abs(change) >= TURN_ANGLE:
        out.append(Maneuver(s, "turn" + TURNS[kind], True, exit_heading, True))
      elif abs(change) >= BEND_ANGLE:
        out.append(Maneuver(s, "keep" + TURNS[kind], False, exit_heading, False))
    else:
      out.append(Maneuver(s, "keep" + FORKS[kind], kind in SIGNALED_FORKS, exit_heading, False))
  return out
