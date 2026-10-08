"""The route's maneuvers the car must do something for (turns and keeps), from its Valhalla maneuvers and shape: the
route input's NEXT and lane slots, and the expert driver's indicators, all read them alike."""
import math

import numpy as np

HEADING_SPAN = 20.0  # m before and after a maneuver for its headings
TURN_ANGLE = 45.0  # deg: a turn; less is a keep
BEND_ANGLE = 15.0  # deg: less than this is straight on
# Valhalla maneuver types
TURNS = {9: "Right", 10: "Right", 11: "Right", 14: "Left", 15: "Left", 16: "Left"}
FORKS = {18: "Right", 19: "Left", 20: "Right", 21: "Left", 23: "Right", 24: "Left"}
SIGNALED_FORKS = {18, 19, 20, 21}  # ramps and exits


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


def maneuvers(route) -> list[Maneuver]:
  out = []
  for m in route.maneuvers:
    kind, i = m.get("type"), m.get("begin_shape_index", 0)
    if kind not in TURNS and kind not in FORKS or i >= len(route.along):
      continue
    s = float(route.along[i])
    before, here, after = (route_point(route, v) for v in (s - HEADING_SPAN, s, s + HEADING_SPAN))
    exit_heading = heading_between(here, after)
    change = wrap(exit_heading - heading_between(before, here))
    if kind in TURNS:
      if abs(change) >= TURN_ANGLE:
        out.append(Maneuver(s, "turn" + TURNS[kind], True, exit_heading, True))
      elif abs(change) >= BEND_ANGLE:
        out.append(Maneuver(s, "keep" + TURNS[kind], False, exit_heading, False))
    else:
      out.append(Maneuver(s, "keep" + FORKS[kind], kind in SIGNALED_FORKS, exit_heading, False))
  return out
