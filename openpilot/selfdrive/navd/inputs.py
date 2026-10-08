"""What navd's planner plans from each step (NavInputs) and what it gives back (NavOutputs).

Map positions are planar metres (x east, y north) about the map's origin, headings degrees counterclockwise from north,
distances along the route metres ahead of the car. The route fields are the router's Route.info. The truth fields are
the simulator's own pose and lane readings, which a real car doesn't have: navd's localizer and lane estimate replace
them, and they go.
"""
from dataclasses import dataclass, field

# requests to the driver, in NavOutputs.requests: signal a turn, change lanes (the driver flicks the stalk, openpilot
# changes lanes on its nudge), or cancel the signal nav asked for
SIGNAL_TURN_LEFT, SIGNAL_TURN_RIGHT = "signalTurnLeft", "signalTurnRight"
LANE_CHANGE_LEFT, LANE_CHANGE_RIGHT = "laneChangeLeft", "laneChangeRight"
CANCEL_SIGNAL = "cancelSignal"


@dataclass
class NavInputs:
  t: float = 0.0  # s, monotonic
  engaged: bool = False
  drive: bool = True  # Navigate on openpilot: nav drives (desires, lane changes, speed caps); off, it only guides
  v: float = 0.0  # m/s (carState.vEgo)
  yaw_rate: float = 0.0  # rad/s, left positive
  blinker: str | None = None  # "left" or "right" while the car's turn signal is on (carState blinkers)
  desire: dict[str, float] = field(default_factory=dict)  # the model's probability of "left", "right", "keepLeft", "keepRight"
  route: list | None = None  # the route ahead from the car, [[x, y], ...] m; empty or None for none
  dest: list | None = None  # [x, y] m, None (or [0, 0]) for none
  route_end: float | None = None  # m ahead to the route's end
  forks: list | None = None  # [[m ahead, side, lanes, lanes_in, keep, other, slip], ...]
  stops: list | None = None  # m ahead of stop lines
  junctions: list | None = None  # m ahead of junction nodes
  limits: list | None = None  # [[m ahead, m/s or 0 unknown], ...]
  lane_arrows: list | None = None  # [[m ahead, [each lane's arrows from the left, ";"-separated]], ...]
  lane_drops: list | None = None  # [[m ahead, first and last lane that carry on, of how many], ...]
  two_way: bool | None = None  # the road here, None unknown
  # truth
  pos: tuple | list = (0.0, 0.0)  # [x, y] m
  heading: float = 0.0  # deg
  truth_lane: list | None = None  # [lane from the left (< 0: oncoming), lanes]
  truth_lane_frac: float | None = None  # the lane, between lanes as it changes
  truth_lane_plugin: list | None = None  # the simulator's own lane reading, as truth_lane
  truth_lane_map: dict | None = None  # the lane by the map's lanes at the true pose: lane, lanes, kind, bay, oncoming, areas


@dataclass
class NavOutputs:
  cap: float  # m/s, 0 for none
  arrived: bool  # stopped at the destination: the destination is done, and the driver disengages
  requests: list[str]  # to the driver, in order
  desires: list[str]  # NavDesire's values set this step, in order ("" clears it)
