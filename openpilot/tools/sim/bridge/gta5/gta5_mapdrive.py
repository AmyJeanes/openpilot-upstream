"""Map driver: expert mode driving our route itself along the map's lanes, rather than the game's AI (gta5_expert.py
with driver=map; design in the openpilot-gta5 task's MAPDRIVER_DESIGN.md). Traffic off for now: no gaps for lane changes,
but other vehicles (the plugin's nearby; traffic off still leaves some) are kept clear of along the path.

The plan is made once when a route arrives (and spliced on a reroute, past the lane change under way and 3 s on):
nav's lane plan (planner.lane_plan, as the ribbon draws it) with absolute along-route keys, each lane change placed
within its lane-slot window (by the trip's style, freeway ones within +-20 % of nav's schedule), then re-placed to take
its time (the style's t_lc, no quicker than its lateral acceleration allows) at the speed the car will drive it, its end
kept, moved off jogs in the route's line (route_jogs, where the lane line swerves; jog_change where there's no
room), and reshaped as a quintic (minimum-jerk) move; then put on the lanes by RouteLanes.lane_line (its offsets, jogs
and fillets), bends the car can't take as drawn eased out (sharp_corner), swinging wide rather than cutting in, and
kept on its lane (_clamp, plan_clamp): our body out of each turn's inside corner kerb (a far-side turn's a median's
edge, else the road's far edge), within its lane outside lane changes and turns' corners, past a fork off its
gore, and off the map's raised islands (traffic_calming=island: a tram crossing's D islands), our body CLAMP_MARGIN
clear of each. That line is the intent, recorded as the path labels; the car aims for it plus a small, slow in-lane bias and
wander (never periodic, faded out through turns, junctions and forks, and within its lane), joining it from where the
car is (on the trip's first plan, a bend it can't take there where the route doubles back fails the plan). Each lane
change, planned or asked for, goes only while its lane is clear by the blind-spot monitor's rules with a lane change's
margins (LaneGapTune: alongside, or closing on us from behind): otherwise it's put off and planned again from the car,
one under way going back to its lane, and where it can't be put off any more the car stops short of it (_lc_gate,
lane_gap events). The plan is
checked against the lane slots' targets (lane_slots.py, compared by where the lanes are): where the slots want another
lane, plan_check=fix (the default) changes into it by the target's end (else as soon after as MIN_LC_M allows, short of
its move), event only logs it, abort ends the trip; each is a plan_slot_mismatch anomaly. A fix into a lane the road
hasn't is left out, and a plan keyed there fails (plan_lane_range).

Each game frame: pure pursuit (lookahead clamp(5, 0.6 v + 3, 25) m) from the rear axle (the game's position is the
car's middle, which moves inwards of its heading in a bend), predicted `latency` (0.1) s on, plus the path's curvature
`ff_preview` (0.32) s ahead as feedforward (the pursuit's own curve-cutting taken out), rate limited to 3x the style's
lateral jerk (more once off the path) and capped at 4.5 m/s^2; the speed follows a static profile (limit or road class x style, the path's
curvature's envelope at the style's lateral acceleration and lateral jerk, nav's turn speeds) with stop signs (full stop and
dwell, then those who came to their lines first let go first: _before_us), give way lines (slow, then on), traffic lights
(no state of their own to read: what the traffic about shows of them, _light, stopped at their line while it's red, come
up to no faster than LIGHT_UNKNOWN_V while there's traffic but no sign; without traffic driven through as before, each
marked light_unknown +-5 s) and a gentle arrival stop_before m short of the end. Other vehicles: one whose body comes within LEAD_SIDE of ours along
the path ahead (in a turn too) is followed (the intelligent driver model's braking, by the style's headway) or stopped
behind, LEAD_STOP short, but a driven one standing across our way (side or head on) is no lead: it's stopped for and
waited for while it moves, the trip ending after DEADLOCK_WAIT s (it may be waiting for us, and the car can't back off);
before a junction or turn the car waits while a moving one's way (predicted from its track, gta5_traffic.py: along
the road where it runs along ours, else on at its yaw rate) crosses ours within CROSS_GAP_AHEAD s after we'd be
through, or CROSS_GAP_BEHIND s before we'd be there, at the junction's signal or give way line where it can still stop
there, else CROSS_STOP short of that way; stopped in a junction it's held where it is (not crept on into the lanes) for the nearest
lane's crosser, one slowed or standing beside our way, and those up to CROSS_T_IN s off, a lapse in them kept
CROSS_KEEP_IN s; and it's no faster than stops for one at the edge of what the
plugin reports (its nearby's "ahead", 15 m where it has none), but for range_share (0.75) of the speed. Pedestrians
(nearby's "p") likewise: one within PED_SIDE of our body along the path is stopped behind, and one walking across it
(anywhere, PED_T s on) waited for, braking however late; where the plugin lists none, one it counts just ahead
(traffic.peds) is braked to a stop for. Static things the plugin's probes hit ahead (nearby's "obst": the map's walls,
kerbs, posts and props in the car's next 15 m) within OBST_SIDE of our body along the path are stopped behind as a
stopped vehicle is, and end the trip after OBST_WAIT s: a plan led into them. Near a junction or turn one within PED_KERB of our body beside the path (waiting
at the kerb, maybe for a light of theirs we can't see), or one anywhere heading for it, caps the speed at what could
still stop short of where they'd step in (ped_kerb). Whether a stop can still be made is judged by what the game
delivers (DECEL_REAL), not what's commanded (DECEL_MAX). A parked vehicle (no one at its wheel, standing), or a driven
one standing across our way, whose body comes within nudge_clear (0.5) m of ours along the path is passed within our own
lane (_nudge): the path shifted away from it to
nudge_want (0.8) m where the lane has room, at least nudge_clear, held from NUDGE_MARGIN before its body to after it and
eased in and out (minimum jerk) over NUDGE_EASE_T s at the speed it's passed at, slower the tighter the pass; our body
kept CLAMP_MARGIN in from the lane's edges (the plan clamp's lane room; where that's no guide, _fill_room's from the
road's sections or our lane's edges carried across a junction: never into a lane the path doesn't take or across the
centre line, and none through a turn). Where there's no room for that past a parked one, or it's seen too late to ease
out, it's passed slowly (BORROW_V) through the room beside our lane (_borrow): our body up to BORROW_LANE into the next
lane our way, BORROW_ONCOMING over the line into the oncoming one (BORROW_WIDE where a narrow lane needs it, with a
wider clear look), or BORROW_EDGE from the road's edge, decided within
BORROW_RANGE of it once that room is clear of others for the pass, else stopped BORROW_STOP short to look again, the trip
ending after BORROW_WAIT s. Where there's no room for even that, it's stopped behind as before. Each is a `nudge` event. Aborts (dev >
1.5 m for 1 s, heading off by 30 deg, a collision, off the road, no progress for 20 s, held by a vehicle for WAIT_MAX s
with none moving (PARKED_WAIT s by a parked one, DEADLOCK_WAIT s by one across our way), lateral acceleration over 4.5,
ONCOMING_S in the oncoming lanes away from a junction (the bridge's lane matching), the driver's input, a failed plan)
brake to a stop and end the trip, for good (on_abort=ai hands it to the game's AI). Map anomalies are logged with their place,
for the map-fix work: kerb_contact (past a road's kerbs, or into a turn's corner kerb, from its legs' cross-sections),
gta_edge (the plan's lane near GTA's road edge where the map's road is wider), lane_disagree, tracking_saturated,
gta_link_offset, stop_line_far, plan_slot_mismatch, plan_lane_range, sharp_corner, jog_change, plan_clamp,
island_contact.

The control file's `mapdrive` object sets it up: {"seed": 7, "preset": "normal", "style": {"t_lc": 5.0}, "bias_max":
0.3, "wander": 0.1, "on_abort": "stop", "speed": null, "plan_check": "fix", "latency": 0.1, "ff_preview": 0.32,
"rear_axle": null, "range_share": 0.75, "nudge": true, "nudge_clear": 0.5, "nudge_want": 0.8, "borrow": true}; bias_max 0
and wander 0 drive exactly on the line; rear_axle (m behind the car's position, default half the state's wheelBase) 0
steers from the car's middle; range_share 1 drops the range cap; nudge false stops behind every parked vehicle in the
way; borrow false keeps every pass within our lane."""
import bisect
import functools
import hashlib
import math
import os
import random
import subprocess

import numpy as np

from openpilot.selfdrive.navd.maneuvers import maneuvers
from openpilot.selfdrive.navd.planner import FREEWAY, TUNE, FwyState, Turn, lane_plan
from openpilot.tools.sim.bridge.gta5.gta5_blindspot import BlindSpotTune, lane_occupied
from openpilot.tools.sim.bridge.gta5.gta5_traffic import ALONG_ANGLE, ALONG_OFF, TURN_OFF, Traffic, body_heading, ctra, rect, size, wrap as wrap_rad
from openpilot.tools.sim.bridge.gta5.gta5_wrongway import project, pursuit

PHASES = ("", "wait", "drive", "stop", "give_way", "arrive", "done", "abort", "follow", "yield")  # gta5.npz mapx phase codes: the index
REASONS = ("", "limit", "curve", "turn", "stop", "give_way", "arrive", "start", "abort", "hold", "lead", "cross", "range",
           "ped", "nudge", "ped_kerb", "blocked", "obstacle", "light", "order")
# lc_tau: 0..1 through the lane change under way by path distance, NaN outside one; lc_dir: its side, -1 left / +1 right
MAPX_COLUMNS = ("phase", "plan_s", "path_dev", "target_lane", "target_right", "v_prof", "a_cmd", "kappa_cmd", "speed_reason",
                "lc_tau", "lead_gap", "lc_dir")
SOURCES = ("none", "ai", "map", "ai_fallback")  # gta5.npz expert_src codes

STEP = 1.0  # m between the path's points
LANE_W = 5.5  # m, GTA's lanes
FRONT = 2.4  # m from the car's origin to its front bumper (and its rear), where the plugin's nearby.dims has none
WHEEL_BASE = 2.8  # m, where the plugin's state has none: the rear axle half of it behind the car's origin
HALF_WIDTH = 1.0  # m, likewise
LATENCY = 0.1  # s from the game's frame to the plugin acting on the control (the pose is predicted this far)
FF_PREVIEW = 0.32  # s ahead the path's curvature is fed forward: the car's curvature lag (about 0.2 s in game) and the delay
LOOKAHEAD = (5.0, 0.6, 3.0, 25.0)  # m: min, s of speed, plus, max
KAPPA_MAX = 0.2  # 1/m
A_LAT_MAX = 4.5  # m/s^2 commanded at most, and an abort measured above it
CURV_SMOOTH = 7.0  # m: the path's curvature averaged over this
FADE_NEAR, FADE_OVER = 15.0, 25.0  # m: in-lane bias and wander gone this near a turn or junction node, back over this
SOFT_IN, SOFT_OUT = 0.5, 1.5  # m an eased corner may come inside the line drawn, and swing outside it before and after
JERK_SHARE = 2.0  # x the style's lateral jerk the speed profile allows where the path's curvature changes
RATE_OFF_PATH = 2.0  # x more of it, 1 m off the path
RATE_SHARE = 3.0  # x it the commanded curvature may change at: a safety limit above the profile's
COMPUTE_EVERY = 0.1  # s: a new frame is computed at once, else at least this often; between, the last control is resent
DECEL_MAX = 4.0  # m/s^2 commanded at most
# m/s^2 the game delivers at most for DECEL_MAX (traffic2 trip 6: -3.2 to -3.4): the plugin clamps the command plus its
# integral on the measured acceleration at -4.0, so the integral has no authority left to make up the shortfall there.
# What can still stop in time is judged by this
DECEL_REAL = 3.2
ABORT_DECEL = 3.5
HOLD_ACCEL = -1.5  # m/s^2 holding a stop
STOPPED = 0.25  # m/s
STOP_NEAR = 2.0  # m from a stop target, stopped: at it
STOP_LAG = 0.3  # s the car carries on at its speed before braking takes hold
GIVE_WAY_SPEED = 3.0  # m/s at the line
LIGHT_MARK = 5.0  # s either side of a light passed without its state
LIGHT_LOOK = 60.0  # m short of a signal's line from which the traffic about is read for its state (_light)
LIGHT_KEEP = 1.5  # s a sign of red or green stands after it was last seen
LIGHT_HOLD_KEEP = 4.0  # s a sign of red stands, held at the line by it
LIGHT_WAIT_MAX = 150.0  # s held at a red light before the trip ends
LIGHT_UNKNOWN_V = 8.0  # m/s a signal's line is come up to at most while the traffic about shows nothing of it
LIGHT_THROUGH = 3.0  # m/s: one of ours crossing its line faster is going through on green
LIGHT_CROSS = 4.0  # m/s: one crossing our way at its junction faster is going on its green
LIGHT_QUEUE = 1.0  # s one of ours has stood at or behind its line, waiting
LIGHT_NODE = 25.0  # m from a signal's junction node a crosser is in its junction
LIGHT_STRAIGHT = 0.15  # rad/s a crosser turns at most, going straight across on its green
ORDER_WAIT = 8.0  # s after its dwell at most a stop sign waits for those who came to their lines first
ORDER_NEAR = 30.0  # m from the junction's node one waiting at its line may be
SIGNAL_TIME, SIGNAL_DIST = 5.0, 50.0  # s, m before a turn (as the AI expert)
TURN_DONE_HEADING, TURN_DONE_AFTER, TURN_PAST, KEEP_PAST = 20.0, 5.0, 40.0, 30.0
JOG_LINK, JOG_TURN, JOG_BACK, JOG_PAD = 25.0, 10.0, 12.0, 10.0  # m, deg, deg, m (route_jogs)
MIN_LC_M = 20.0  # m a lane change takes at the least
SPLICE_BACK = 150.0  # m short of where a new route leaves the old one's roads the old path is kept to
SAME_LINE = 1.5  # m between two routes' lines on the same roads
SPLICE_GAP = 0.3  # m between the kept path's end and the new one closed as a lane change
# an arrow key's turn (route.prompt) keeps the old path 3 s on as any new route, but only to PROMPT_ROOM_S s of driving
# (PROMPT_ROOM m at the least) short of where it turns off, and PROMPT_KEEP_S s at the least: its junction may be near
PROMPT_KEEP_S = 1.0
PROMPT_ROOM_S = 3.0
PROMPT_ROOM = 30.0
ASK_WAIT = 5.0  # s a lane change the player asks for waits for the lane beside to be clear, at most
ASK_KEEP_S = 10.0  # s of driving (and ASK_KEEP m at the least) nav mustn't want back out of the lane asked for within
ASK_KEEP = 150.0
ASK_PHASES = ("drive", "follow")
LC_GATE_T = 1.0  # s before a lane change starts from which its lane is to be clear (_lc_gate)
# share of the way across after which a change is carried through (before it, gone back from): our middle still in
# our lane, so the monitor's lane beside is the one moved into (past it, the lane beyond that); going back later takes
# longer than letting one closing on us brake
LC_COMMIT = 0.25
LC_DEFER_S, LC_DEFER_MIN = 2.0, 15.0  # s on from the car (m at the least) a change put off starts again
LC_LEAST_SHARE = 0.6  # of a change's time (lc_seconds) one put off keeps at the least
LC_SETTLE_S = 1.0  # s in its lane between a change put off and the next one of its run, at the least
LC_CLEAR_S = 0.5  # s its lane is to read clear, after it was busy, before a change waited for starts
LC_AFTER_S = 1.5  # s after a change ends its lane is still looked at for one coming up on us (_lane_window)
LC_GAP_M, LC_GAP_T, LC_GAP_CLOSE = 2.0, 0.3, 1.0  # m, s of the follower's speed and s of its closing speed a change leaves
LC_CUT_DECEL = 2.0  # m/s^2 a change may leave us to slow at behind one in its lane
LC_SLOW, LC_GAP_SLOW = 3.0, 1.0  # m/s both are slower than, and m a change leaves then
BRANCH_ROOM_S = 3.0  # s of driving at least before a fork, for its lane change given up to take the other branch
BRANCH_PAST = 20.0  # m past a fork the link of the branch given up that a route is asked to keep off
LC_LATE_WAIT = 90.0  # s stopped short of a change that can't be put off, its lane busy, before the trip ends (a queue standing in it moves on at its light)
LC_DROP_BACK = 0.75  # share of its speed as it began to wait the car slows to while one keeps alongside in that lane


class LaneGapTune(BlindSpotTune):
  """The lane beside clear for a lane change of ours: the blind-spot monitor's zone across (the lane beside), from
  `behind` m behind our rear bumper (behind_s s of our speed, within behind_min..behind) to `front` m ahead of our
  front, and anyone in that lane behind it closing on us faster than closing_speed due within closing_time s, as far
  back as the plugin reports. In slow, dense traffic a car length's gap is a gap, as GTA's drivers let one in."""
  DEFAULTS = {**BlindSpotTune.DEFAULTS, "behind": 10.0, "behind_s": 0.5, "behind_min": 3.0, "front": 1.0,
              "closing_behind": 45.0, "closing_speed": 1.0, "closing_time": 3.0}
LC_A_LAT = 0.6  # x the style's lateral acceleration, a lane change's peak at most
LC_MAX_S = 9.0  # s a change takes at the most
LC_SHORTEST = 12.0  # m a change re-timed to its time takes at the least, a lane (pulling away, slowly)
LC_FIT = 0.15  # a change timed more than this share off its time (by the speed it'll be driven at) is re-placed
LC_MIN_SHARE = 0.6  # of its time a change squeezed by what's before it keeps at least, else it's left as it was
JOIN_MAX = 8.0  # m from its lane line the car may start
BEND_DEG = 20.0  # deg a route point turns that makes its road beside it unclear
KAPPA_DRIVABLE = 0.1  # 1/m: a tighter bend in the lane line (a jog, a corner without a fillet) is eased out
LC_DONE = 0.8  # share of a lane change after which its signal goes off
CROSSING_CLEAR = 5.0  # m past a stop line or junction node a city lane change may start
LANE_RANGE_AHEAD = 60.0  # m on from a key the road's most lanes are the lanes it may number (_lanes_about)
STEP_SPAN = 0.1  # m between a plan's keys that are steps of one renumbering (_outside_lanes)
TURN_SPAN = 10.0  # m either side of a turn's point held to its turn speed
# vehicles (the plugin's nearby: their bodies in our frame, read every 0.1 s)
LEAD_SIDE = 0.4  # m beside our body another's may come along the path before it's in the way
LEAD_AHEAD = 80.0  # m of path looked along
LEAD_OFF = 6.0  # m off the path at most a body's point is placed along it
LEAD_STOP = 4.0  # m from our front bumper to a stopped vehicle we stop at
# static things the plugin's probes hit (nearby's "obst": points, each taken as a body OBST_HALF m either way)
OBST_SIDE = 0.25  # m beside our body one may be along the path before it's in the way (kerbs run alongside lanes)
OBST_HALF = 0.1  # m
OBST_WAIT = 3.0  # s stopped behind one before the trip ends
LEAD_BESIDE = 1.0  # m behind our front bumper a body may start and still be ahead (not beside us)
NEARBY_AHEAD = 15.0  # m ahead of our origin the plugin reports vehicles, where its nearby has no "ahead"
NEARBY_SIDE = 15.0  # m either side, likewise ("side")
ALONG_BEND = 15.0  # m of path about a vehicle's place whose bends it may be taking (_predict)
SPEED_UP_MAX, SPEED_UP_TO = 3.0, 15.0  # m/s^2 and m/s one speeding up is taken to go on to, looking behind for a lane change
RANGE_LAG = 0.6  # s before braking takes hold, for the range cap (the command's delay, its jerk and the car's lag)
RANGE_STOP = 1.0  # m short of a vehicle at the range's end the range cap stops at
RANGE_BODY = 2.5  # m of its body before its middle (which the plugin's reach is to)
WAIT_MAX = 60.0  # s held by a vehicle in the way (none moving) before the trip ends
PARKED_WAIT = 10.0  # s, by a parked one (no one in its driver's seat)
DEADLOCK_WAIT = 20.0  # s, by a driven one standing across our way (it may be waiting for us), after backing off it
REVERSE_AFTER = 6.0  # s held, stopped, by a driven one standing across our way close ahead before backing off it
REVERSE_NEAR = 8.0  # m from our front bumper it stands within, for that
REVERSE_DIST = 3.0  # m backed off
REVERSE_V = 1.0  # m/s backed off at
REVERSE_T = 6.0  # s backing off at most
REVERSE_WAIT = 4.0  # s stood still after, for it to go by, before going on
REVERSE_MAX = 2  # times backed off a trip, at most
REVERSE_BEHIND = 4.0  # m behind where the back-off ends that's to be clear of anyone (now and REVERSE_T s on)
ACROSS_ANGLE = 35.0  # deg off our way a standing vehicle's heading is, at least, for it to be across our way
LEAD_MOVING = 0.5  # m/s: a vehicle in the way going faster is moving on
CROSS_MOVING = 2.0  # m/s: a slower vehicle isn't crossing
CROSS_T = 6.0  # s ahead a crossing vehicle's way is predicted (straight on at its speed)
CROSS_GAP_AHEAD = 1.2  # s from our rear clearing its way to its body reaching it, for us to go first
CROSS_GAP_BEHIND = 0.6  # s from its body clearing our way to our front reaching it, for us to go after it
CROSS_START_LAG = 0.5  # s a standing start takes to get going, more than the acceleration alone
CROSS_ANGLE = 25.0  # deg off our way at least, for a crossing
CROSS_NEAR = 25.0  # m from a junction node or turn the crossing may be, to be held for
CROSS_STOP = 2.0  # m short of its way we stop at
CROSS_KEEP = 0.5  # s a hold is kept after the crossing last looked likely
CROSS_KEEP_IN = 1.5  # s, stopped in a junction (a crosser flickering at the edge of what's looked at isn't a gap)
CROSS_SLOW = 3.0  # m/s we're slower than, in a junction, for its crossing holds (_crossing)
CROSS_T_IN = 10.0  # s ahead a crossing vehicle's way is predicted then: time to clear the junction from a standstill
CROSS_BESIDE = 1.0  # m beside our body a slowed crosser may stand, in a junction, to be held for
SLOWED_TURN = 15.0  # m of path on from its place along which our way may turn to its heading, for it to be no crosser
JUNCTION_IN = 25.0  # m past a stop line our front bumper is in its junction for
JUNCTION_NODE = 10.0  # m from a junction node, likewise
# pedestrians (the plugin's nearby "p": their places and velocities, a body PED_HALF m either way of each)
PED_HALF = 0.3  # m
PED_SIDE = 1.0  # m beside our body a pedestrian's may come before it's in the way (they step about)
PED_MOVING = 0.5  # m/s: a slower pedestrian isn't crossing
PED_T = 4.0  # s ahead a crossing pedestrian's way is predicted
PED_BOX = (2.5, 1.0, 12.0)  # m: the plugin's traffic.peds counts those within this either side of our origin, this far ahead to this
PED_KERB = 4.0  # m beside our body a pedestrian near a crossing, or heading for our path, is made ready for
PED_KERB_DECEL = 3.0  # m/s^2 the car could stop short of where one would step into the path with (after RANGE_LAG)
PED_KERB_MIN = 6.0  # m/s the cap holds at the least, beside them (a step out then is a crossing's, braked for at once)
PED_STEP = 2.0  # m/s a pedestrian may step towards the path at: one we'd be past before they could reach it is no matter
# passing a parked vehicle within our lane (_nudge)
NUDGE_CLEAR = 0.5  # m our body keeps from its passing it, at the least; above LEAD_SIDE, so once shifted it's no lead
NUDGE_WANT = 0.8  # m, where the lane has room
NUDGE_MARGIN = 2.0  # m before and after its body the shift is held
NUDGE_EASE_T = 2.0  # s a shift eases in and out over at the least, at the speed it's passed at
NUDGE_EASE_MIN = 8.0  # m: a shift with less room to ease in or out isn't made
NUDGE_A_LAT = 0.5  # x the style's lateral acceleration an ease's peak at most (a quintic's is 5.77 shift / length^2 x v^2)
NUDGE_V = (6.0, 20.0)  # m/s passing at NUDGE_CLEAR, and more for each m clearer
NUDGE_LATE = 1.3  # x the speed an ease is made for the car may still be doing where it starts, braking from now
NUDGE_GUARD = 1.0  # m ahead of the car within which the path is never moved
NUDGE_SAME = 1.0  # m between a parked vehicle's readings that are one
NUDGE_NODE = 12.0  # m from a junction node, or short of a turn's point, within which the map's sections don't guide a nudge
NUDGE_ALIGN = 10.0  # deg off its road's line the path may head where the road's section guides a nudge
NUDGE_SPAN = (2.6, 7.5)  # m a lane's width is within where it guides a nudge
NUDGE_BRIDGE = 50.0  # m of road without a guide a nudge's room is carried across, on a straight
NUDGE_BRIDGE_DEG = 40.0  # deg the path may bend across it (less than a turn)
NUDGE_FILL = 60.0  # m either side of a shift's hold its room is filled in over
NUDGE_ACROSS = 40  # m of path either side the lanes it moves across between are taken as ours
# past a parked vehicle our lane alone has no room past, slowly through the room beside it (_borrow)
BORROW_LANE = 2.0  # m our body may come into the next lane our way
BORROW_ONCOMING = 1.0  # m over the line into an oncoming or centre lane
BORROW_WIDE = 2.0  # m over it where a narrow lane leaves no less, the room looked across and ahead the wider (_borrow_clear)
BORROW_EDGE = 0.1  # m our body keeps in from the road's edge (CLAMP_MARGIN within a lane)
BORROW_KINDS = ("", "edge", "lane", "oncoming")  # bolder on
BORROW_PREFER = ("lane", "oncoming", "edge")  # the side a pass takes where both have room
BORROW_MARGIN = 1.0  # m before and after its body the shift is held
BORROW_KAPPA = 0.18  # 1/m an ease turns no tighter than (a quintic's peak curvature is 5.77 shift / length^2)
BORROW_EASE, BORROW_EASE_MIN = 15.0, 4.0  # m an ease takes at most, and at the least
BORROW_V = 4.0  # m/s the pass is made at, at most
BORROW_RANGE = 25.0  # m from our front to it within which the pass is decided (on what's about then)
BORROW_RETRY = 0.5  # s between looks while the room is busy
BORROW_STOP = 12.0  # m short of it we stop at meanwhile: room to pull out round it
BORROW_WAIT = 30.0  # s held behind it, the room busy, before the trip ends (PARKED_WAIT where there's none)
BORROW_BEHIND, BORROW_AHEAD = 10.0, 10.0  # m behind our rear and past the pass's end the room is looked along
BORROW_SIDE, BORROW_SIDE_ONCOMING = 0.5, 1.0  # m beside our body it's looked across, and more into the oncoming lanes
BORROW_SPARE = 2.0  # s more than the pass takes at BORROW_V that no one may come into the room within
# aborts
DEV_ABORT, DEV_ABORT_S = 1.5, 1.0
HEADING_ABORT, HEADING_ABORT_S = 30.0, 0.5
NO_PROGRESS_S = 20.0
A_LAT_ABORT_S = 0.5
ONCOMING_S = 1.0  # s in the oncoming lanes (state.laneMap), away from a junction
OFF_ROAD = 1.0  # m past the kerb
OFF_ROAD_DEV = 0.75  # m off the path as well, for an off-road abort
USER_STEER = 0.02
# anomalies
KERB_MARGIN = 0.2  # m the car's side may come past the kerb before it counts
CORNER_LEG = 25.0  # m before and after a turn's point its legs' kerbs are taken, outside the junction
CORNER_ANGLE = (30.0, 150.0)  # deg a turn checked against its corner kerb turns
CORNER_LEG_MIN = 6.0  # m: a corner nearer the next than twice this has no legs of its own to take
CORNER_LEGS_SHORT = (15.0,)  # m: shorter legs tried where the road bends away into the block beyond the first
CORNER_SAME = 10.0  # m between a turn and a corner of the lane line that are one
MEDIAN_GAP = 1.0  # m between the two directions that's a median
MEDIAN_RADIUS = 2.0  # m a median's nose is taken as rounded
CORNER_NODE = 20.0  # m from the turn's point to a junction node, for a junction's corner
CORNER_NEAR = 20.0  # m from the turn's point the kerb lines' corner may be
CORNER_INSIDE = 0.5  # m the plan keeps inside each leg's kerb line, or the line isn't the kerb
CORNER_SPAN = 40.0  # m either side of a turn's point checked against its corner kerb
KERB_RADIUS = 6.0  # m: a junction's corner kerbs taken as rounded this much (the map has no corner geometry)
FAR_RADIUS = 0.5  # m a far-side corner taken as rounded: a road's far edge ends at whatever stands there (a parapet's end)
BODY_STEP = 0.2  # m between the points along our body's sides tested against a corner kerb
EDGE_WIDER = 0.75  # m the map's road edge may lie past GTA's before a lane near GTA's edge is suspect
EDGE_CLEAR = 0.8  # m from the car's side to GTA's road edge (its link's lanes) below which the plan is too near it
JUNCTION_CLEAR = 25.0  # m from a junction node, where cross-sections don't describe the road, kerbs aren't checked
LANE_DISAGREE_S = 2.0
SATURATED_S = 0.5
GTA_OFFSET = 1.5  # m between GTA's lane centre and the map's for the planned lane
STOP_LINE_FAR = 30.0  # m from a stop line on to its junction's first node
ANOMALY_SPACING = 20.0  # m between anomalies of one kind
# the plan kept on its lane (_clamp)
CLAMP_MARGIN = 0.3  # m our body keeps in from a corner's kerb or its lane's edges
CLAMP_TURN = 45.0  # deg a turn turns, about whose point the lane's edges are left to the corner kerbs
CORNER_ZONE = 15.0  # m either side of such a turn's point
CLAMP_EASE = 15.0  # m a push eases in and out over
LANE_GRID = 2.0  # m between the places the lane is read at
CLAMP_MAX = 2.5  # m a push is at most
# m a raised island pushes the plan at most: a plan clipping one. One needing more runs through its middle (a route
# across a median, a lane laid over it that nav didn't move from), which a push only bends into a sharper S over it:
# that island is left to the island_contact anomaly and the plugin's obstacle probes
ISLAND_MAX = 2.5
ISLAND_ROOM = 1.0  # m beside our body (and CLAMP_MARGIN) an island keeps the lane clamp from pulling the plan back over it
CARRY_MAX = 50.0  # m without a guide of its own a lane is carried across, at most (_lane_room)
FORK_BEFORE, FORK_AFTER = 20.0, 40.0  # m before and after a fork its gore side is kept to
FORK_ROOM = 0.3  # m from the lane's centre towards the gore at most
# road class speeds (m/s) where the map has no limit: 65, 55, 40, 35, 30, 25, 15 mph
CLASS_SPEED = {"motorway": 29.1, "trunk": 24.6, "motorway_link": 17.9, "trunk_link": 17.9, "primary": 15.6, "secondary": 15.6,
               "tertiary": 13.4, "primary_link": 13.4, "secondary_link": 13.4, "tertiary_link": 11.2, "residential": 11.2,
               "unclassified": 11.2, "living_street": 6.7, "service": 6.7}
DEFAULT_SPEED = 13.4

# the trip's style: (low, high, which way "brisk" goes: +1 high, -1 low, 0 neither), drawn once per trip from its seed
STYLE_RANGES = {
  "speed": (0.85, 1.10, 1), "a_lat": (1.5, 2.8, 1), "j_lat": (1.2, 2.5, 1), "accel": (1.0, 2.0, 1), "decel": (1.0, 2.0, 1),
  "jerk": (1.0, 2.0, 1), "turn": (0.9, 1.15, 1), "t_lc": (4.0, 6.5, -1), "lc_timing": (0.1, 0.9, 0),
  "fwy_timing": (-0.2, 0.2, 0), "signal_lead": (1.5, 4.0, -1), "stop_margin": (0.5, 2.5, -1), "stop_dwell": (0.5, 2.0, -1),
  "reaction": (0.4, 1.5, -1), "bias": (-1.0, 1.0, 0), "headway": (1.0, 2.0, -1),
}
PRESETS = {"calm": 0.2, "normal": 0.5, "brisk": 0.8}
DEFAULTS = {"seed": None, "preset": "normal", "style": {}, "bias_max": 0.3, "wander": 0.1, "wander_m": 300.0,
            "on_abort": "stop", "speed": None, "plan_check": "fix", "stop_before": 15.0, "hold_after": 3.0, "retime": True,
            "drive_on_right": True, "latency": LATENCY, "ff_preview": FF_PREVIEW, "fit_durations": True,
            "rear_axle": None, "range_share": 0.75, "nudge": True, "nudge_clear": NUDGE_CLEAR, "nudge_want": NUDGE_WANT,
            "borrow": True}


def pick_style(seed: int, preset: str = "normal", over: dict | None = None) -> dict:
  """A trip's style from its seed: each value drawn about the preset's place in its range (brisk: the quick end)."""
  rng = random.Random(seed)
  centre = PRESETS.get(preset, 0.5)
  out = {}
  for k, (lo, hi, way) in STYLE_RANGES.items():
    u = rng.random() if way == 0 else min(max(rng.gauss(centre if way > 0 else 1.0 - centre, 0.15), 0.0), 1.0)
    out[k] = round(lo + (hi - lo) * u, 3)
  out.update(over or {})
  return out


_VERSION: dict | None = None


def map_version() -> dict:
  """The map's and the code's versions, for recordings: the sha1 of the map's gta5.osm.pbf (GTA5_MAP) and the git rev."""
  global _VERSION
  if _VERSION is None:
    _VERSION = {"pbf_sha1": None, "git": None}
    pbf = os.path.join(os.getenv("GTA5_MAP", os.path.expanduser("~/gta5map_lanes")), "gta5.osm.pbf")
    try:
      h = hashlib.sha1()
      with open(pbf, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
          h.update(block)
      _VERSION["pbf_sha1"] = h.hexdigest()[:16]
    except OSError:
      pass
    try:
      _VERSION["git"] = subprocess.check_output(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "--short=12", "HEAD"],
                                                text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
      pass
  return _VERSION


# *** the plan's keys ***

EPS = 1e-3


def lane_at(keys: list[tuple[float, float]], s: float) -> float:
  """The keys' lane s m along: the last key at or before s (the later of two at one place), towards the next."""
  i = bisect.bisect_right([k[0] for k in keys], s) - 1
  if i < 0:
    return keys[0][1]
  if i >= len(keys) - 1:
    return keys[-1][1]
  (s0, l0), (s1, l1) = keys[i], keys[i + 1]
  return l1 if s1 - s0 < EPS else l0 + (l1 - l0) * (s - s0) / (s1 - s0)


def _ramp(keys, j: int) -> bool:
  return keys[j + 1][0] - keys[j][0] > EPS and abs(keys[j + 1][1] - keys[j][1]) > EPS


def chains(keys: list[tuple[float, float]]) -> list[tuple[int, int]]:
  """Each lane change in the keys: (first key, last key), a run of ramps with any renumbering steps (two keys at one
  place, where the road's lanes change) inside it. Two ramps meeting in a lane's centre where it doesn't renumber
  (keys alike), or turning back the other way, are two changes, one after the other."""
  out, i, n = [], 0, len(keys)
  while i < n - 1:
    if not _ramp(keys, i):
      # a change can begin with a jump in its numbering where it starts (nav's ramp across a lane map at its start)
      if not (i + 2 < n and keys[i + 1][0] - keys[i][0] <= EPS and abs(keys[i + 1][1] - round(keys[i + 1][1])) > 0.05 and
              _ramp(keys, i + 1) and (keys[i + 2][1] - keys[i + 1][1]) * (keys[i + 1][1] - keys[i][1]) > 0):
        i += 1
        continue
    j = i
    way = keys[i + 1][1] - keys[i][1]

    def same_way(m: int) -> bool:
      return (keys[m + 1][1] - keys[m][1]) * way > 0
    def joined(m: int) -> bool:  # keys m and m + 1 at one place, inside a change: renumbered, or not at a lane's centre
      return keys[m + 1][0] - keys[m][0] <= EPS and (abs(keys[m + 1][1] - keys[m][1]) > EPS or
                                                     abs(keys[m][1] - round(keys[m][1])) > 0.05)
    while j < n - 1 and ((_ramp(keys, j) and same_way(j)) or
                         (joined(j) and j + 1 < n - 1 and _ramp(keys, j + 1) and same_way(j + 1))):
      j += 1
    out.append((i, j))
    i = j
  return out


def quintic(t):
  t = np.clip(t, 0.0, 1.0)
  return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


def shaped(keys: list[tuple[float, float]], step: float = 2.0) -> list[tuple[float, float]]:
  """The keys with each lane change a quintic (minimum-jerk) move across, sampled every `step` m, in place of a ramp;
  renumbering steps inside a change stay where they are, at their share of the move."""
  out: list[tuple[float, float]] = []
  last = 0
  for a, b in chains(keys):
    out += keys[last:a]
    s0, s1 = keys[a][0], keys[b][0]
    span = max(s1 - s0, EPS)
    for j in range(a, b):
      (sa, la), (sb, lb) = keys[j], keys[j + 1]
      if sb - sa <= EPS:
        continue
      d = (lb - la) / (sb - sa) * span  # the whole move's lanes in this numbering
      x = la - d * (sa - s0) / span
      ss = np.unique(np.concatenate((np.arange(sa, sb, step), [sb])))
      out += [(float(s), float(x + d * quintic((s - s0) / span))) for s in ss]
    last = b + 1
  out += keys[last:]
  dedup = [out[0]] if out else []
  for k in out[1:]:
    if abs(k[0] - dedup[-1][0]) > EPS or abs(k[1] - dedup[-1][1]) > EPS:
      dedup.append(k)
  return dedup


def lc_seconds(style: dict, lanes: float) -> float:
  """s a change across `lanes` lanes takes: the style's (more lanes by their square root), and no quicker than a peak
  lateral acceleration of LC_A_LAT x the style's allows (a quintic's peak is 5.77 D / T^2)."""
  n = max(abs(lanes), 1.0)
  a = LC_A_LAT * style["a_lat"]
  return float(min(max(style["t_lc"] * math.sqrt(n), math.sqrt(5.77 * LANE_W * n / a)), LC_MAX_S))


# *** the path ***

def resample(points: np.ndarray, step: float = STEP) -> tuple[np.ndarray, np.ndarray]:
  p = np.asarray(points, float)[:, :2]
  keep = np.concatenate(([True], np.hypot(*np.diff(p, axis=0).T) > 1e-6))
  p = p[keep]
  s = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))))
  if len(p) < 2:
    return p, s
  su = np.arange(0.0, s[-1], step)
  su = np.append(su, s[-1]) if s[-1] - su[-1] > 0.25 * step else su
  return np.stack([np.interp(su, s, p[:, 0]), np.interp(su, s, p[:, 1])], axis=1), su


def headings(p: np.ndarray) -> np.ndarray:
  """Radians counterclockwise from east, unwrapped, at each point."""
  d = np.gradient(p, axis=0)
  return np.unwrap(np.arctan2(d[:, 1], d[:, 0]))


def smooth(x: np.ndarray, n: int) -> np.ndarray:
  if n <= 1 or len(x) < n:
    return x
  pad = n // 2
  xp = np.concatenate((np.full(pad, x[0]), x, np.full(pad, x[-1])))
  return np.convolve(xp, np.ones(n) / n, mode="valid")[:len(x)]


def wander(s: np.ndarray, rng: random.Random, sigma: float, spacing: float) -> tuple[np.ndarray, list[float]]:
  """Slow smooth lateral wander: random knots every `spacing` m (normal, sigma m), eased between (C1, flat at each
  knot), so it never weaves at a fixed period."""
  if sigma <= 0 or not len(s):
    return np.zeros_like(s), []
  n = int(s[-1] // spacing) + 2
  knots = [rng.gauss(0.0, sigma) for _ in range(n)]
  k = np.minimum((s // spacing).astype(int), n - 2)
  t = s / spacing - k
  e = t * t * (3 - 2 * t)
  kn = np.array(knots)
  return kn[k] + (kn[k + 1] - kn[k]) * e, [round(v, 3) for v in knots]


@functools.lru_cache(maxsize=16)
def _outline(front: float, rear: float, side: float) -> np.ndarray:
  """Points round a body's outline (m forward of its origin, m left), its corners and at most BODY_STEP apart."""
  xs = np.linspace(-rear, front, int(math.ceil((front + rear) / BODY_STEP)) + 1)
  ys = np.linspace(-side, side, int(math.ceil(2 * side / BODY_STEP)) + 1)[1:-1]
  return np.vstack([np.stack([xs, np.full_like(xs, side)], axis=1), np.stack([xs, np.full_like(xs, -side)], axis=1),
                    np.stack([np.full_like(ys, front), ys], axis=1), np.stack([np.full_like(ys, -rear), ys], axis=1)])


def soften(p: np.ndarray, kmax: float, iters: int = 400) -> tuple[np.ndarray, list[tuple[int, float]]]:
  """The polyline (points STEP apart) with bends tighter than kmax eased out, as a driver cuts a corner the car can't
  take as drawn (a lane line's jog, a corner without a fillet): their points pulled towards their neighbours'
  middle until none is. Also each such bend as it was: (point index, its curvature)."""
  p = np.asarray(p, float).copy()

  def bends(q):
    s = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(q, axis=0).T))))
    k = smooth(np.gradient(headings(q), np.maximum(s, np.arange(len(s)) * 1e-6)), 3)
    bad = np.abs(k) > kmax
    bad[:2] = bad[-2:] = False
    return k, bad
  if len(p) < 5:
    return p, []
  k, bad = bends(p)
  e = np.flatnonzero(np.diff(np.concatenate(([0], bad.astype(np.int8), [0]))))
  spots = [(int((a + b) // 2), float(k[a:b][np.argmax(np.abs(k[a:b]))])) for a, b in zip(e[::2], e[1::2], strict=True)]
  for a, b in zip(e[::2], e[1::2], strict=True):
    # each bend in a window around it, the window's ends held
    lo, hi = max(int(a) - 60, 0), min(int(b) + 60, len(p))
    w = p[lo:hi].copy()
    orig = w.copy()
    for _ in range(iters):
      _, bw = bends(w)
      bw[:3] = bw[-3:] = False
      if not bw.any():
        break
      m = np.convolve(bw.astype(float), np.ones(9), mode="same") > 0
      m[:2] = m[-2:] = False
      mid = w.copy()
      mid[1:-1] = 0.5 * (w[:-2] + w[2:])
      w[m] += 0.5 * (mid[m] - w[m])
    # easing a corner cuts inside it, towards the kerb's corner: no more than SOFT_IN, the curve moved out instead
    # (up to SOFT_OUT, eased in and out over the window), as a driver swings wide for a corner too tight to take
    peak = float(k[a:b][np.argmax(np.abs(k[a:b]))])
    th = headings(orig)
    inward = np.sign(peak) * np.stack([-np.sin(th), np.cos(th)], axis=1)  # towards the corner's centre
    cut = np.einsum("ij,ij->i", w - orig, inward)
    excess = min(float(cut.max()) - SOFT_IN, SOFT_OUT)
    if excess > 0:
      n = len(w)
      x = np.arange(n) / max(n - 1, 1)
      centre = (int((a + b) // 2) - lo) / max(n - 1, 1)
      u = np.clip(np.where(x < centre, x / max(centre, 1e-6), (1 - x) / max(1 - centre, 1e-6)) * 1.6, 0.0, 1.0)
      w -= inward * (excess * u * u * (3 - 2 * u))[:, None]
    p[lo:hi] = w
  return p, spots


def hermite_join(path: np.ndarray, pos: np.ndarray, heading: float, v: float, tol: float = 8.0) -> np.ndarray:
  """The path's start replaced by a cubic from the car's pose (heading in game deg) into it, where the car heads more
  than `tol` deg off the path's way."""
  if len(path) < 8:
    return path
  j = int(min(max(12.0, 2.5 * v) / STEP, len(path) - 3))
  th = headings(path[:3])[0]
  near = float(np.min(np.hypot(*(path[:j] - np.asarray(pos, float)).T)))
  if abs(wrap(heading - game_heading(th))) <= tol and near < 0.75:
    return path
  p0, p1 = np.asarray(pos, float), path[j]
  d = float(np.hypot(*(p1 - p0)))
  h = math.radians(heading)
  t0 = np.array([-math.sin(h), math.cos(h)]) * d
  t1 = (path[j + 1] - path[j - 1]) / max(float(np.hypot(*(path[j + 1] - path[j - 1]))), 1e-9) * d
  u = np.linspace(0.0, 1.0, j + 1)[:, None]
  curve = (2 * u**3 - 3 * u**2 + 1) * p0 + (u**3 - 2 * u**2 + u) * t0 + (-2 * u**3 + 3 * u**2) * p1 + (u**3 - u**2) * t1
  return np.vstack([curve[:-1], path[j:]])


def route_jogs(route) -> list[tuple[float, float]]:
  """Where the route's line jogs sideways and back (a short link turning JOG_TURN or more one way, then back within
  JOG_BACK of the way it was going, as across a junction where a road's line moves over to its other carriageway's):
  [(m along from, to)], padded JOG_PAD either side. The lane line swerves there, by the lanes' offsets from a diagonal."""
  p = route.points[:, :2]
  if len(p) < 4:
    return []
  d = np.diff(p, axis=0)
  h = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
  seg = np.hypot(*d.T)
  out = []
  for k in range(1, len(d) - 1):
    if seg[k] > JOG_LINK or seg[k] < 1e-3:
      continue
    t_in, t_out = wrap(h[k] - h[k - 1]), wrap(h[k + 1] - h[k])
    if abs(t_in) >= JOG_TURN and abs(t_out) >= JOG_TURN and t_in * t_out < 0 and abs(t_in + t_out) < JOG_BACK:
      out.append((float(route.along[k]) - JOG_PAD, float(route.along[k + 1]) + JOG_PAD))
  return out


def along_route(route, pts: np.ndarray, start: float) -> tuple[np.ndarray, np.ndarray]:
  """Each point's (m along the route, m right of its line), searched on from `start` m along so where the route
  passes back near itself the points stay on their own part."""
  rp, ra = route.points[:, :2], route.along
  k = max(int(np.searchsorted(ra, start, side="right")) - 1, 0)
  out_s, out_r = np.empty(len(pts)), np.empty(len(pts))
  batch = 32
  for n0 in range(0, len(pts), batch):
    p = pts[n0:n0 + batch]
    # the segments from a little behind the last point's to as far on as the batch could reach
    lo = max(k - 1, 0)
    hi = min(max(int(np.searchsorted(ra, ra[k] + batch * STEP * 1.5 + 30.0)), lo + 2), len(rp) - 1)
    a, ab = rp[lo:hi], rp[lo + 1:hi + 1] - rp[lo:hi]
    ab2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9)
    rel = p[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum("nij,ij->ni", rel, ab) / ab2, 0.0, 1.0)
    d = np.hypot(*(rel - ab[None] * t[..., None]).transpose(2, 0, 1))
    i = np.argmin(d, axis=1)
    n = np.arange(len(p))
    seg = np.sqrt(ab2[i])
    out_s[n0:n0 + batch] = ra[lo + i] + t[n, i] * seg
    out_r[n0:n0 + batch] = (rel[n, i, 0] * ab[i, 1] - rel[n, i, 1] * ab[i, 0]) / seg
    k = lo + int(i[-1])
  return np.maximum.accumulate(out_s), out_r


def segment_of(route, s: float) -> int:
  return int(min(max(np.searchsorted(route.along, s, side="right") - 1, 0), max(len(route.points) - 2, 0)))


def road_speed(route, k: int) -> float:
  """The road's speed on route segment k: the map's limit, else by its class."""
  limits = getattr(route, "limits", None)
  if limits is not None and k < len(limits) and limits[k] > 0:
    return float(limits[k])
  classes = getattr(route, "classes", None) or []
  return CLASS_SPEED.get(classes[k] if k < len(classes) else "", DEFAULT_SPEED)


def wrap(deg: float) -> float:
  return (deg + 180) % 360 - 180


def game_heading(theta: float) -> float:
  """The game's heading (deg counterclockwise from north) of a direction theta radians from east."""
  return wrap(math.degrees(theta) - 90.0)


class MapDrive:
  """One trip's drive along the map, which the map driver (gta5_drivers.MapDriver) drives with. step() each bridge step returns the plugin control message (resent between frames);
  `phase`, `events` (since last taken), `indicator`/`label` (the desire it signals) and `finished` (None, "arrived" or
  "abort: <why>") for expert mode."""
  def __init__(self, cfg: dict | None = None, tune=None):
    self.c = {**DEFAULTS, **(cfg or {})}
    seed = self.c.get("seed")
    self.seed = int(seed) if seed is not None else random.randrange(1 << 31)
    self.style = pick_style(self.seed, str(self.c["preset"]), self.c.get("style") or {})
    self.rng = random.Random(self.seed + 1)
    self.tune = tune or TUNE
    self.phase = ""
    self.events: list[dict] = []
    self.finished: str | None = None
    self.why = ""
    self.route = None
    self.plans = 0
    self.splices: list[dict] = []
    self.anomalies: list[dict] = []
    self.aborts: list[dict] = []
    self.lights: list[dict] = []
    self.plan: dict | None = None  # plan_record of the plan driven
    self.indicator: str | None = None
    self.label = "none"
    self.reason = ""
    self.msg: dict | None = None
    self.last_key = None
    self.last_t: float | None = None
    self.t0: float | None = None
    self.collisions0: int | None = None
    self.kappa = 0.0
    self.a = 0.0
    self.s = 0.0
    self.dev = 0.0
    self.v_prof = math.nan
    self.lc_tau = math.nan
    self.lc_dir = math.nan
    self.target_lane = math.nan
    self.target_right = math.nan
    self.timers: dict[str, float | None] = {}
    self.progress = (0.0, 0.0)  # (s, t) of the last progress
    self.stopped_at: float | None = None
    self.arrived_t: float | None = None
    self.engage0: int | None = None
    self._last_anomaly: dict[str, float] = {}
    self.path = None
    self.join_m = 0.0  # m of path from the car into its lane line, where the road's own checks wait
    self.corners: list[dict] = []
    self.clamps: list[dict] = []  # where _clamp moved the plan back: {"i": path index, "push": m right, "why", "length"}
    self.jog_changes: list[tuple[float, float]] = []  # lane changes left across a jog, with no room off it
    self.moves_along: list[float] = []  # m along the route of the lane slots' moves (turns, forks, junctions through)
    self.unclear = np.zeros(0)
    self.front, self.rear, self.half_width = FRONT, FRONT, HALF_WIDTH  # our car's body from its origin (_body)
    self.cars: list[dict] | None = None  # the vehicles around, in the world (_vehicles); None: the plugin reports none
    self.cars_rows = None
    self.traffic = Traffic()  # their tracks, from report to report
    self.peds_reported = False  # whether the plugin's nearby lists pedestrians ("p")
    self.reach = (NEARBY_AHEAD, NEARBY_SIDE)  # m ahead of our origin and to either side the plugin reports them
    self.lead_gap = math.nan  # m from our front bumper to the vehicle in our way along the path
    self.lead_right = math.nan  # m right of the path its body comes nearest
    self.lead_driven: bool | None = None  # whether anyone drives it
    self.held_since: float | None = None  # when a vehicle in the way began holding the car up
    self.cross_hold: tuple | None = None  # (path m we stop at, when, details) of the last crossing held for
    self.lane_room = None  # _clamp's _lane_room of the plan
    self.path0 = None  # the path without the nudges (_derive's)
    self.nudge_off = np.zeros(0)  # m right of path0 the path is moved at each point
    self.nudge_fixed = np.zeros(0)  # of it, the shifts passed
    self.room_lo = self.room_hi = np.zeros(0)  # how far (m right) each point may move within its lane; NaN: no guide
    self.borrow_lo = self.borrow_hi = np.zeros(0)  # how far past its lane, where a pass (_borrow) was made; NaN: none
    self.nudge_groups: list[dict] = []  # the shifts made (_group), until passed
    self.nudge_blocked: list[dict] = []  # parked vehicles with no shift past them, until passed
    self.nudge_log: list[dict] = []  # each nudge event, for the summary
    self.kerb_logged = -math.inf  # when the pedestrian kerb cap last held the car back
    self.obst: list[dict] = []  # static things ahead the plugin's probes hit, as bodies in the world (_obstacles)
    self.obst_rows = None
    self.gap_tune = LaneGapTune()
    self.lc_wait: dict | None = None  # the lane change waiting for its lane (_lc_gate): since, side, defers, late, s0
    self.last_state: dict = {}
    self.asked: dict | None = None  # the lane change the player asked for, waiting for the lane beside: side, t, waited
    self.prompts: list[tuple[str, str]] = []  # (kind, side) of what became of it, since last taken (ask_lane)
    self.last_v = 0.0  # m/s at the last step
    self.reversing: dict | None = None  # backing off a deadlock: when it began, path m it ends at, when it stopped
    self.avoid: list | None = None  # links (pa, pb) a route is to keep off, asked for once (take_avoid): a fork's branch given up
    self.reverses = 0  # times backed off

  def _body(self, state: dict):
    """Our car's body from the plugin's nearby.dims (its model's bounds: min x, max x, min y, max y)."""
    dims = (state.get("nearby") or {}).get("dims")
    if dims and len(dims) == 4 and dims[3] > 0.5 and dims[2] < -0.5 and dims[1] - dims[0] > 0.5:
      self.front, self.rear, self.half_width = float(dims[3]), float(-dims[2]), float(max(-dims[0], dims[1]))

  @property
  def body(self) -> tuple[float, float, float]:
    return self.front, self.rear, self.half_width

  # *** what expert mode and recordings read ***

  def info(self) -> dict:
    return {"phase": self.phase, "s": round(self.s, 1), "dev": round(self.dev, 2), "lane": None if math.isnan(self.target_lane) else round(self.target_lane, 2),
            "vProf": None if math.isnan(self.v_prof) else round(self.v_prof, 2), "a": round(self.a, 2), "kappa": round(self.kappa, 5),
            "reason": self.reason, "lcTau": None if math.isnan(self.lc_tau) else round(self.lc_tau, 2), "label": self.label,
            "lead": None if math.isnan(self.lead_gap) else round(self.lead_gap, 1),
            "leadRight": None if math.isnan(self.lead_right) else round(self.lead_right, 2), "leadDriven": self.lead_driven,
            "nudge": self._nudge_now()}

  def _nudge_now(self) -> dict | None:
    """The shift under way where the car is: m right of the path without it now, and the clearance it was made for."""
    g = next((g for g in self.nudge_groups if g["start"] <= self.s <= g["end"]), None)
    if g is None:
      return None
    return {"off": round(float(np.interp(self.s, self.s_path, self.nudge_off)), 2),
            "clear": round(min(e["clear"] for e in g["entries"]), 2)}

  def mapx_row(self) -> list[float]:
    return [PHASES.index(self.phase) if self.phase in PHASES else 0, self.s, self.dev, self.target_lane, self.target_right, self.v_prof,
            self.a, self.kappa, REASONS.index(self.reason) if self.reason in REASONS else 0, self.lc_tau, self.lead_gap, self.lc_dir]

  def summary(self) -> dict:
    """gta5.json's mapx: the trip's style and seed, its plans' keys, splices, aborts, anomalies and lights passed."""
    return {"seed": self.seed, "style": self.style, "cfg": {k: v for k, v in self.c.items() if k != "style"}, "version": map_version(),
            "keys": self.plan["keys"] if self.plan else None, "plans": self.plans, "splices": self.splices, "aborts": self.aborts,
            "anomalies": self.anomalies, "lights": self.lights, "slot_check": self.plan["slot_check"] if self.plan else None,
            "nudges": self.nudge_log}

  def intent_line(self, step: int = 2) -> list:
    """The intent (the planned path without the in-lane wander) from the car on, for the ribbon and recordings."""
    if self.path is None:
      return []
    i = max(int(self.s / STEP) - 2, 0)
    return self.intent[i::step].round(1).tolist()

  def take_prompts(self) -> list[tuple[str, str]]:
    p, self.prompts = self.prompts, []
    return p

  def take_avoid(self) -> list | None:
    """The links a new route should keep off (a fork's branch whose lane change couldn't be made in time), once: the
    driver's navigator routes again with them."""
    out, self.avoid = self.avoid, None
    return out

  def take_events(self) -> list[dict]:
    e, self.events = self.events, []
    return e

  def _event(self, what: str, when: float, **kw):
    self.events.append({"event": what, "t": round(when, 3), **kw})

  def _set(self, phase: str, t: float, **kw):
    if phase != self.phase and self.phase != "abort":  # an abort ends the trip: nothing leaves it
      self.phase = phase
      self._event("mapdrive", t, phase=phase, **kw)

  def _anomaly(self, kind: str, when: float, pos, at: float, **kw):
    last = self._last_anomaly.get(kind)
    if last is not None and abs(at - last) < ANOMALY_SPACING:
      return
    self._last_anomaly[kind] = at
    a = {**kw, "kind": kind, "t": round(when, 3), "pos": [round(float(pos[0]), 1), round(float(pos[1]), 1)], "at": round(at, 1)}
    self.anomalies.append(a)
    self._event("anomaly", when, **{k: v for k, v in a.items() if k != "t"})

  def _abort(self, why: str, t: float, pos=None):
    if self.phase == "abort":
      return
    self.why = why
    self.aborts.append({"why": why, "t": round(t, 3), "s": round(self.s, 1), "pos": None if pos is None else [round(float(pos[0]), 1), round(float(pos[1]), 1)]})
    self._set("abort", t, why=why)

  # *** planning ***

  def _speed_at(self, route, s: float) -> float:
    v = road_speed(route, segment_of(route, s)) * self.style["speed"]
    return min(v, float(self.c["speed"])) if self.c.get("speed") else v

  def nav_keys(self, route, lane) -> list[tuple[float, float]]:
    """nav's lane plan from the car's lane, as the ribbon has it, with absolute along-route keys."""
    info = route.info(route.length)
    v = self._speed_at(route, route.at)
    keys = lane_plan(route.rest(), info["forks"], lane, route.lanes_at, v, self.tune, info["laneArrows"], info["laneDrops"],
                     maps=route.lane_maps(route.length), turns=route.turns(route.length),
                     classes=route.changes(route.classes, route.length), limits=route.changes(route.limit_list, route.length),
                     fwy=FwyState(), crossings=[a - route.at for a in route.stops + route.junctions if a > route.at])
    out, last = [], -math.inf
    for d, lane in keys:  # in order along: nav's can step back a few cm
      last = max(last, route.at + d)
      out.append((last, float(lane)))
    return out

  def retime(self, route, keys: list[tuple[float, float]], slots) -> list[tuple[float, float]]:
    """Each single lane change over the style's time at the road's speed, its end placed by the style within its
    move's lane-slot window (freeway ones nav's schedule +-fwy_timing of a lane's time), clear of stop lines and junction
    nodes in the city, never begun back across the move before; one from where its lane opens kept there, no shorter
    than MIN_LC_M; a city one nav squeezed under MIN_LC_M that finds no room so is run before the crossing it's on
    (_room_before)."""
    keys = list(keys)
    crossings = sorted(route.stops + route.junctions)
    for a, b in chains(keys):
      if b != a + 1:
        continue  # renumbering inside it: left as nav has it
      (s0, la), (s1, lb) = keys[a], keys[b]
      if a > 0 and s0 - keys[a - 1][0] <= EPS:
        # from where its lane opens: as nav has it, but no shorter than MIN_LC_M where that ends short of the next
        # crossing and key (nav squeezes it between the opening and the target's end)
        least = MIN_LC_M * max(abs(lb - la), 1.0)
        room = min([keys[b + 1][0] - 1.0 if b + 1 < len(keys) else math.inf] + [c - CROSSING_CLEAR for c in crossings if c > s1])
        if s1 - s0 < least <= room - s0:
          keys[b] = (float(s0 + least), lb)
        continue
      v = self._speed_at(route, s0)
      length = max(lc_seconds(self.style, lb - la) * v, MIN_LC_M * max(abs(lb - la), 1.0))
      j = a - 1  # back over holds in its lane to the change, step or start before it
      while j > 0 and abs(keys[j - 1][1] - keys[j][1]) < EPS and keys[j][0] - keys[j - 1][0] > EPS:
        j -= 1
      lower = keys[j][0] if a > 0 else s0
      back = lower
      upper = keys[b + 1][0] if b + 1 < len(keys) else s1
      window = None
      if slots is not None:
        for m in slots.moves:
          if m.along >= s1 - 1.0 and m.need:
            lower = max(lower, min(m.free + CROSSING_CLEAR, s0))  # not back across the move before (its lanes may not lead on)
            _, w0, w1 = m.window(v)
            if w1 >= s1 - 1.0:
              window = (w0, w1)
            break
      k = segment_of(route, s0)
      fwy = (route.classes[k] if k < len(route.classes) else "") in FREEWAY
      hi = min(upper, window[1] if window is not None else s1)
      if fwy:
        e = s1 + self.style["fwy_timing"] * self.tune.fwy_lane_time * v
      elif window is not None:
        lo = max(window[0], lower) + length
        e = lo + self.style["lc_timing"] * max(hi - lo, 0.0)
      else:
        e = s1
      e = min(max(e, lower + length), hi)
      start = max(e - length, lower)
      placed = e - start >= 0.6 * length  # else no room: as nav has it
      if placed and not fwy:
        for c in crossings:
          if start - CROSSING_CLEAR < c < e:
            if c + CROSSING_CLEAR + length <= hi:
              start, e = c + CROSSING_CLEAR, c + CROSSING_CLEAR + length
            elif c - CROSSING_CLEAR - length >= lower:
              start, e = c - CROSSING_CLEAR - length, c - CROSSING_CLEAR
        if any(start - CROSSING_CLEAR < c < e for c in crossings) and not any(s0 - CROSSING_CLEAR < c < s1 for c in crossings):
          placed = False  # only nav's timing was clear of them
      placed = placed and lower - EPS <= start < e <= upper + EPS
      least = MIN_LC_M * max(abs(lb - la), 1.0)
      if not placed and not fwy and s1 - s0 < least - EPS:
        # nav had no room for it (a lane ending just past a junction squeezes it onto the junction's node)
        before = self._room_before(route, slots, crossings, s0, s1, lb, back, length, least)
        if before is not None:
          (start, e), placed = before, True
      if placed:
        keys[a], keys[b] = (float(start), la), (float(e), lb)
    return keys

  def _room_before(self, route, slots, crossings: list[float], s0: float, s1: float, lane: float, back: float,
                   length: float, least: float) -> tuple[float, float] | None:
    """Where a city lane change into `lane` that nav squeezed into s0..s1 may run instead: ending CROSSING_CLEAR short
    of the stop line or junction node it's on (else where nav began it), `length` long or as long as there's room for
    down to `least`, not before `back` nor across a crossing, nor back across a move whose slot lanes don't include
    `lane`; None where there's no such room."""
    on = [c for c in crossings if s0 - CROSSING_CLEAR < c <= s1 + EPS]
    e = min(on) - CROSSING_CLEAR if on else s0
    start = max([e - length, back] + [c + CROSSING_CLEAR for c in crossings if c < e])
    if e - start < least:
      return None
    for m in slots.moves if slots is not None else []:
      if m.need and start < m.along <= s1 + 1.0:
        good, target = self._slot_lanes(route, slots, m.along - 2.0, self._speed_at(route, m.along))
        if good is None or target.move is not m or int(round(lane)) not in good:
          return None
    return start, e

  def slot_check(self, route, keys: list[tuple[float, float]], slots) -> dict:
    """The plan against the lane slots' targets: from each target's end to its move, the plan's lane (rounded) must be
    one of them, compared by where the lanes are (the slots number the lanes as they are open, the plan as the road's
    lanes carry on)."""
    out = {"checked": 0, "mismatches": []}
    if slots is None:
      return out
    for m in slots.moves:
      if not m.need or m.along <= route.at + 5.0:
        continue
      v = self._speed_at(route, m.along)
      _, _, end = m.window(v)
      # where the lanes renumber between (a bay or lane opening, which nav moves into only once it's there), from a
      # lane change's length past it
      steps = [keys[i][0] for i in range(len(keys) - 1) if keys[i + 1][0] - keys[i][0] <= EPS and
               abs(keys[i + 1][1] - keys[i][1]) > EPS and end - EPS <= keys[i][0] < m.along - 1.0]
      first = max([end] + [x + MIN_LC_M for x in steps] + ([m.opens + MIN_LC_M] if m.opens is not None else []))
      if end <= m.free + EPS:  # no room before the move before: the target's end is a lane change past it
        first = max(first, m.free + CROSSING_CLEAR + MIN_LC_M)
      bad = None
      for s in np.arange(max(first, route.at + 1.0), m.along - 2.0, 2.0):
        good, target = self._slot_lanes(route, slots, float(s), v)
        if not good or target.move is not m:
          continue
        out["checked"] += 1
        lane = lane_at(keys, float(s))
        if int(round(lane)) not in good and abs(lane - round(lane)) < 0.25:
          bad = {"s": round(float(s), 1), "move": round(m.along, 1), "end": round(float(end), 1), "plan": round(lane, 2),
                 "want": sorted(good), "slots": sorted(target.want)}
          break
      if bad is not None:
        out["mismatches"].append(bad)
    return out

  @staticmethod
  def _slot_lanes(route, slots, s: float, v: float):
    """The slots' target lanes s m along as the plan numbers them (RouteLanes.section_at's lanes whose centres lie in
    a target lane as the slots have them, opened_at's), and the target; (None, target) without lanes to aim for."""
    here, target = slots.target(s, v)
    if target is None or target.centre or not target.want or here is None:
      return None, target
    sec = route.lanes.section_at(s)
    if sec is None or not sec.lanes:
      return None, target
    spans = [here.ours[w] for w in target.want if w < len(here.ours)]
    good = {i for i in range(sec.lanes) if any(sp.left - 0.3 <= sec.offset(i) <= sp.right + 0.3 for sp in spans)}
    return good or None, target

  def follow_slots(self, route, keys: list[tuple[float, float]], mismatches: list[dict], slots) -> list[tuple[float, float]]:
    """The keys with a lane change into the slots' nearest target lane for each mismatch, done by its target's end
    (from a lane change's length before, not over the plan's change before nor before the move or turn before, whose
    target the plan keeps through it), else as soon after as MIN_LC_M allows, short of its move; or where the plan's own
    change for the move goes past the target lane, that change stopped in it; then in that lane to its move:
    the plan's own changes there dropped, its renumbering steps (lanes beginning or ending) kept. Past the move, the plan
    on from the lane it's then in (a change of its own under way there counted from where it began), but from a turn's
    own renumbering step there as nav has it, as that says which lane the turn comes out in."""
    keys = list(keys)
    moves = slots.moves if slots is not None else []
    for bad in sorted(mismatches, key=lambda b: b["s"]):
      end = min(bad["end"], bad["s"])
      move = next((m.along for m in moves if abs(m.along - bad["move"]) < 0.11), bad["move"])  # not slot_check's rounding
      v = self._speed_at(route, end)
      good, _ = self._slot_lanes(route, slots, end + 0.1, v)
      want = sorted(good) if good else bad["want"]
      p = round(lane_at(keys, end + 0.1))
      if p in want:  # still in a target lane at the end, out of it by the move: a change of its own under way there
        p = round(lane_at(keys, bad["s"]))
      w = min(want, key=lambda x: abs(x - p))
      delta = w - p
      if delta == 0 or abs(lane_at(keys, move - 0.5) - w) < 0.02:
        continue  # in it by the move already: a change a fix before finished after the target's end
      length = max(lc_seconds(self.style, delta) * v, MIN_LC_M * abs(delta))
      changes = [keys[b][0] for a, b in chains(keys) if keys[b][0] <= end]
      # (a turn whatever its lanes, and one at the target's end too: a window with no room before it ends there)
      before = [m.along + CROSSING_CLEAR for m in moves if (m.need or m.turn) and m.along < end + 0.11 and m.along < move - 1.0]
      # nor before the lane it's into has opened
      shut = [x for x in np.arange(move - 1.0, end - length - LANE_GRID, -LANE_GRID)
              if (sec := route.lanes.opened_at(float(x))) is not None and sec.lanes and w >= sec.lanes]
      start = max(end - length, max(changes + before + [x + LANE_GRID for x in shut[:1]], default=keys[0][0]), keys[0][0])
      # the plan's own change for this move (after the move before, done by the target's end) past the target lane: it
      # stops in it instead, rather than a change back after it, late, across the stop line
      own = [(a, b) for a, b in chains(keys) if keys[b][0] <= end + EPS]
      a, b = own[-1] if own else (0, 0)
      moved = self._chain_lanes(keys, a, b) if own and keys[a][0] >= max(before, default=-math.inf) - EPS else 0.0
      if abs(moved) > EPS and 0.0 < (keys[b][1] - w) / moved <= 1.0 + EPS:
        out = keys[:a] + self._retarget_chain(keys, a, b, float(w))
        end = keys[b][0]
      else:
        if end - start < 0.4 * length:
          end = min(start + 0.4 * length, move - 1.0)
          if end - start < MIN_LC_M * abs(delta):
            continue  # no room: left as it is (still a mismatch)
        out = [k for k in keys if k[0] <= start]
        out += [(start, out[-1][1] if out else lane_at(keys, start)), (end, float(w))]  # a change under way there dropped
      lane, i = float(w), 0
      inside = [k for k in keys if end < k[0] < move - EPS]
      while i < len(inside):
        s, a = inside[i]
        j = i
        while j + 1 < len(inside) and inside[j + 1][0] - s <= EPS:
          j += 1
        if j > i:  # keys at one place: a renumbering step, carried
          b = inside[j][1]
          out += [(s, lane), (s, lane + (b - a))]
          lane += b - a
        i = j + 1  # else a point of the plan's own change or hold: the lane stays
      # past the move, the plan on from the lane it's now in: its own changes towards that lane already made (the gap
      # taken up by them), renumbering steps carried
      rest = [k for k in keys if k[0] >= move - EPS]
      pv = lane_at(keys, move - 0.01)
      if abs(pv - round(pv)) > 0.02:  # a change of the plan's own under way at the move: from the lane it began in
        on = next((val for s, val in rest if s > move + EPS), pv)
        pv = float(math.ceil(pv) if on < pv else math.floor(pv))
      gap = lane - pv
      turn = any(m.turn and abs(m.along - move) < 1.0 for m in moves)
      at = len(out)
      out.append((move, lane))
      for n, (s, val) in enumerate(rest):
        step = n > 0 and s - rest[n - 1][0] <= EPS and abs(val - rest[n - 1][1]) > EPS
        if step and turn and s < move + 1.0:
          gap = 0.0
        dv = val - pv
        if not step and abs(gap) > EPS and dv * gap > 0:
          gap -= math.copysign(min(abs(dv), abs(gap)), gap)
        lanes = self._lanes_about(route, rest, n)
        if lanes and not -0.5 < val + gap < lanes - 0.5:  # our lane ends there (the road narrows): in the nearest
          gap = min(max(val + gap, 0.0), lanes - 1.0) - val
        out.append((s, val + gap))
        pv = val
      # the rest of a change of its own that began before the move: no shorter than MIN_LC_M
      a = next((a for a, b in chains(out) if a == at and b == a + 1), None)
      least = MIN_LC_M * max(abs(out[a + 1][1] - out[a][1]), 1.0) if a is not None else 0.0
      if a is not None and out[a + 1][0] - out[a][0] < least:
        room = out[a + 2][0] - 1.0 if a + 2 < len(out) else math.inf
        out[a + 1] = (max(out[a + 1][0], min(out[a][0] + least, room)), out[a + 1][1])
      keys = out
    return keys

  @staticmethod
  def _outside_lanes(route, keys: list[tuple[float, float]]) -> dict | None:
    """The first key whose lane lies outside the lanes the road has for it (_lanes_about): {"s", "lane", "lanes"};
    None where all are inside."""
    for i, (s, lane) in enumerate(keys):
      lanes = MapDrive._lanes_about(route, keys, i)
      if lanes and not -0.5 < lane < lanes - 0.5:
        # nav renumbers across a sliver of a section in steps a few cm apart, none of them driven: the road's lanes
        # just before the first step count for them all
        a = i
        while a > 0 and keys[a][0] - keys[a - 1][0] <= STEP_SPAN:
          a -= 1
        before = MapDrive._lanes_about(route, [(keys[a][0] - 0.1, keys[a][1])], 0) if a < i else 0
        if before and -0.5 < lane < before - 0.5:
          continue
        return {"s": round(s, 1), "lane": round(lane, 2), "lanes": lanes}
    return None

  @staticmethod
  def _lanes_about(route, keys: list[tuple[float, float]], i: int) -> int:
    """How many lanes the plan's key i may number: the road's most within LANE_RANGE_AHEAD m on (nav numbers a lane
    about to open as it will be), from just before a renumbering step for the key before it, just after for the key
    after; 0 where the road's lanes are unknown."""
    s = keys[i][0]
    if i + 1 < len(keys) and keys[i + 1][0] - s <= EPS:
      s -= 0.05
    elif i > 0 and s - keys[i - 1][0] <= EPS:
      s += 0.05
    sec = route.lanes.section_at(s, segment_of(route, s))
    if sec is None or not sec.lanes:
      return 0
    ahead = [route.lanes.section_at(x, segment_of(route, x)) for x in s + np.arange(10.0, LANE_RANGE_AHEAD + 1.0, 10.0)]
    return max([sec.lanes] + [a.lanes for a in ahead if a is not None])

  def _make_plan(self, route, state: dict, t: float, shift: int = 0) -> bool:
    """The plan from the car; with `shift` (-1 left, 1 right) nav's from the lane beside, after a change into it the
    player asked for (_asked_keys)."""
    lanes = route.lanes
    if lanes is None:
      self._abort("plan: no lanes on the route", t, state.get("pos"))
      return False
    lane = route.lane() if route.on_road() else None
    if lane is None or lane[0] < 0:
      sec = route.section(route.seg)
      if sec is None or not sec.lanes:
        k = next((j for j in range(route.seg, min(route.seg + 6, len(lanes.sections))) if lanes.sections[j] is not None and lanes.sections[j].lanes), None)
        sec = lanes.sections[k] if k is not None else None
      if sec is None:
        self._abort("plan: no lanes where it starts", t, state.get("pos"))
        return False
      lane = [min(max(sec.lane(route.right), 0), sec.lanes - 1), sec.lanes]
    try:
      from openpilot.selfdrive.navd.lane_slots import LaneSlots
      slots = LaneSlots(route, bool(self.c["drive_on_right"]))
    except Exception as e:  # the slots on a map they can't read: plan without them
      self._event("mapdrive_warning", t, why=f"lane slots: {e!r}")
      slots = None
    self.moves_along = sorted(m.along for m in slots.moves) if slots is not None else []
    nav = self.nav_keys(route, [lane[0] + shift, lane[1]] if shift else lane)
    keys = self.retime(route, nav, slots) if self.c["retime"] else nav
    if shift:
      keys = self._asked_keys(route, keys, lane, float(state.get("vEgo") or 0.0))
    check = self.slot_check(route, keys, slots)
    fixed = []
    for _ in range(4 if self.c["plan_check"] == "fix" else 0):  # a fix can leave the next move wanting another lane
      if not check["mismatches"]:
        break
      tried = self.follow_slots(route, keys, check["mismatches"], slots)
      if tried == keys:
        break  # no room for any of them
      outside = self._outside_lanes(route, tried)
      if outside is not None:  # a fix into a lane the road hasn't (the oncoming one): the plan as it was
        self._anomaly("plan_lane_range", t, route_point(route, outside["s"]), outside["s"], fix=True, **outside)
        break
      fixed += check["mismatches"]
      keys = tried
      check = {**self.slot_check(route, keys, slots), "fixed": fixed}
    outside = self._outside_lanes(route, keys)
    if outside is not None:
      self._anomaly("plan_lane_range", t, route_point(route, outside["s"]), outside["s"], fix=False, **outside)
      self._abort(f"plan: lane {outside['lane']} {outside['s']} m along, where the road has {outside['lanes']}", t, state.get("pos"))
      return False
    for bad in fixed:
      self._anomaly("plan_slot_mismatch", t, route_point(route, bad["s"]), bad["s"], fixed=True, **bad)
    for bad in check["mismatches"]:
      self._anomaly("plan_slot_mismatch", t, route_point(route, bad["s"]), bad["s"] + 0.5, fixed=False, **bad)
    if check["mismatches"] and self.c["plan_check"] == "abort":
      self._abort(f"plan: lane {check['mismatches'][0]['plan']} where the slots want {check['mismatches'][0]['want']}", t, state.get("pos"))
      return False
    at0 = route.at
    built = self._build(route, keys, lane, state, t)
    if built is None:
      return False
    fitted = self.fit_durations(route, keys, float(state.get("vEgo") or 0.0)) if self.c["fit_durations"] else keys
    fitted = self.off_jogs(route, fitted)
    if fitted != keys:
      keys = fitted
      built = self._build(route, keys, lane, state, t)
      if built is None:
        return False
    intent, sharp, knots, bias = built
    for i, k in sharp:
      self._anomaly("sharp_corner", t, intent[min(i, len(intent) - 1)], i * STEP, kappa=round(k, 3))
    for c in self.clamps:
      i = min(c["i"], len(intent) - 1)
      self._anomaly("plan_clamp", t, intent[i], i * STEP, why=c["why"], push=c["push"], length=round(c["length"], 1))
    self.keys, self.nav_plan = keys, nav
    self.plans += 1
    self.plan = {"at0": round(at0, 2), "keys": [(round(s, 2), round(lane_, 3)) for s, lane_ in keys],
                 "nav_keys": [(round(s, 2), round(lane_, 3)) for s, lane_ in nav], "lane": list(lane), "bias": round(bias, 3),
                 "wander_knots": knots, "wander_m": float(self.c["wander_m"]) / 2, "slot_check": check,
                 "stops": [[round(x["along"], 1), x["kind"]] for x in self.stops], "path": self.intent[::2].round(2).tolist()}
    self.changes = self._changes(keys)  # each lane change on the path: (path m from, to, side, m across)
    self._plan_anomalies(route, t)
    return True

  def _asked_keys(self, route, keys: list[tuple[float, float]], lane, v: float) -> list[tuple[float, float]]:
    """keys (nav's from the lane beside) after a change into it from the car's lane, signal_lead s on."""
    s0 = route.at + self.style["signal_lead"] * max(v, 3.0)
    s1 = s0 + max(lc_seconds(self.style, 1.0) * max(v, 5.0), MIN_LC_M)
    return [(route.at, float(lane[0])), (s0, float(lane[0])), (s1, keys[0][1])] + [k for k in keys if k[0] > s1 + EPS]

  def _build(self, route, keys: list[tuple[float, float]], lane, state: dict, t: float):
    """The path for keys from the car (_derive's): (intent, sharp bends eased, wander knots, bias), None (an abort)
    where there's no line."""
    lanes = route.lanes
    dense = shaped(keys)
    at0 = route.at
    line = lanes.lane_line(at0, [(s - at0, lane_) for s, lane_ in dense if s >= at0 - 1e-6] or [(0.0, float(lane[0]))])
    if line is None or len(line) < 3:
      self._abort("plan: no lane line", t, state.get("pos"))
      return None
    intent, s_path = resample(np.asarray(line, float))
    if len(intent) < 3:
      self._abort("plan: lane line too short", t, state.get("pos"))
      return None
    sharp: list = []
    for n in range(4):  # a near reversal can take more than one pass
      intent, found = soften(intent, KAPPA_DRIVABLE)
      intent, s_path = resample(intent)
      sharp = found if n == 0 else sharp
      if not found:
        break
    intent, room_l, room_r = self._clamp(route, keys, intent)
    intent, s_path = resample(intent)
    bias = self.style["bias"] * float(self.c["bias_max"])
    off, knots = wander(s_path, random.Random(self.seed + 1), float(self.c["wander"]), float(self.c["wander_m"]) / 2)
    n = min(len(off), len(room_l))
    off[:n] = np.clip(bias + off[:n], -room_l[:n], room_r[:n])  # no further than its lane leaves room for
    off = np.clip(off, -0.45, 0.45) * np.clip(s_path / 20.0, 0.0, 1.0) * self._fade(route, intent)
    # from where the car is, across to the line as a lane change would (setup places it in a lane, a reroute may not):
    # by the route's own line where the car is on it (past a start stub, RouteLanes.start), as the line's start can
    # swerve (a fillet, a clamp)
    sec = lanes.section_at(at0, segment_of(route, at0))
    if route.on_road() and not route.elsewhere and at0 >= lanes.start and sec is not None and sec.lanes:
      e0 = float(route.right - sec.offset(lane_at(dense, at0)))
    else:
      e0 = self._offset_from(intent, np.asarray(state["pos"][:2], float))
    if abs(e0) > JOIN_MAX:
      self._abort(f"plan: the car is {e0:+.1f} m from its lane line", t, state.get("pos"))
      return None
    v0 = max(float(state.get("vEgo") or 0.0), 5.0)
    join = max(MIN_LC_M, lc_seconds(self.style, e0 / LANE_W) * v0) if abs(e0) > 0.5 else 15.0
    near = [i * STEP for i, _ in sharp if i * STEP < join]
    # a bend where it pulls away that the route's own line doubles back into (a link back down the road from the car's
    # place): the lane line is wrong there; a junction's corner near the start is a real bend, driven eased
    if near and not self.plans and self._reverses(route, at0 + near[-1] + 5.0):
      self._abort(f"plan: a sharp corner {near[0]:.0f} m on, where the car joins its line", t, state.get("pos"))
      return None
    off = off + e0 * (1.0 - quintic(s_path / join))
    self.join_m = max(join, 30.0)
    self._derive(route, intent, off, t, start=(np.asarray(state["pos"][:2], float), float(state.get("heading") or 0.0), v0))
    return intent, sharp, knots, bias

  @staticmethod
  def _reverses(route, upto: float) -> bool:
    """Whether the route's line turns back on itself (more than 90 deg between two links) within `upto` m along."""
    k = int(np.searchsorted(route.along, upto, side="right")) + 1
    d = np.diff(route.points[:k, :2], axis=0)
    d = d[np.hypot(*d.T) > 1e-3]
    return bool(len(d) > 1 and ((d[:-1] * d[1:]).sum(axis=1) < 0).any())

  def _fade(self, route, intent: np.ndarray) -> np.ndarray:
    """1 along the line, easing to 0 within FADE_NEAR m of the route's turns, junction nodes, forks and sharp bends,
    where the car keeps to the line itself."""
    along, _ = along_route(route, intent, route.at - 5.0)
    d = np.diff(route.points[:, :2], axis=0)
    h = np.arctan2(d[:, 1], d[:, 0])
    bend = np.degrees(np.abs((np.diff(h) + np.pi) % (2 * np.pi) - np.pi))
    spots = np.concatenate((np.asarray(route.junctions, float), route.along[1:-1][bend > BEND_DEG],
                            [route.at + tr[0] for tr in route.turns(route.length)], [f.along for f in route.forks]))
    if not len(spots):
      return np.ones(len(intent))
    spots = np.sort(spots)
    i = np.clip(np.searchsorted(spots, along), 1, len(spots) - 1) if len(spots) > 1 else np.zeros(len(along), int)
    near = np.minimum(np.abs(spots[i] - along), np.abs(spots[np.maximum(i - 1, 0)] - along))
    u = np.clip((near - FADE_NEAR) / FADE_OVER, 0.0, 1.0)
    return u * u * (3 - 2 * u)

  def drive_speeds(self, v0: float) -> np.ndarray:
    """The speed the car will drive the path at (m/s, at least 1): the static profile, stopping at stop signs, and
    no quicker up to it than the style's acceleration from v0 (as from a standstill, or after a stop)."""
    v = self.v_static.copy()
    sp = self.s_path
    b, a = self.style["decel"] * 0.8, self.style["accel"] * 0.8
    for st in self.stops:
      if st["kind"] == "stop":
        i = int(np.clip(np.searchsorted(sp, st["target"]), 0, len(v) - 1))
        v[i] = 0.0
        for j in range(i - 1, -1, -1):
          lim = math.sqrt(v[j + 1] ** 2 + 2 * b * (sp[j + 1] - sp[j]))
          if lim >= v[j]:
            break
          v[j] = lim
    v[0] = min(v[0], max(v0, 0.0))
    for j in range(len(v) - 1):
      v[j + 1] = min(v[j + 1], math.sqrt(v[j] ** 2 + 2 * a * (sp[j + 1] - sp[j])))
    return np.maximum(v, 1.0)

  def fit_durations(self, route, keys: list[tuple[float, float]], v0: float) -> list[tuple[float, float]]:
    """The keys with each lane change re-placed to take its time (lc_seconds) at the speed the car will drive it
    (drive_speeds of the path just built), its end kept: started later where it'd be slow, earlier where quick, never
    before the key before it nor back across the move before; city ones not across a stop line or junction node it
    didn't cross already. Where that
    leaves it short and it's the move into the plan's lanes from the car's (at the plan's start), it ends later
    instead, up to the key after. Renumbering steps inside a change stay at their places."""
    original, keys = keys, list(keys)
    v = self.drive_speeds(v0)
    sp = self.s_path
    tcum = np.concatenate(([0.0], np.cumsum(np.diff(sp) / np.maximum(0.5 * (v[1:] + v[:-1]), 1.0))))
    crossings = sorted(route.stops + route.junctions)

    def path_s(along):
      return float(np.interp(along, self.route_s, sp))

    def along_of(ps):
      return float(np.interp(ps, sp, self.route_s))

    def secs(s_a, s_b):
      return float(np.interp(path_s(s_b), sp, tcum) - np.interp(path_s(s_a), sp, tcum))

    def at_time(s_from, dt):
      return along_of(float(np.interp(np.interp(path_s(s_from), sp, tcum) + dt, tcum, sp)))

    for a, b in reversed(chains(keys)):  # from the last, so earlier indices hold as keys come and go
      (s0, la), (s1, _) = keys[a], keys[b]
      lanes = abs(self._chain_lanes(keys, a, b))
      want = lc_seconds(self.style, lanes)
      if abs(secs(s0, s1) - want) <= LC_FIT * want:
        continue
      j = a - 1  # back over holds in its lane to the change, step or start before it
      while j > 0 and abs(keys[j - 1][1] - keys[j][1]) < EPS and keys[j][0] - keys[j - 1][0] > EPS:
        j -= 1
      lower = keys[j][0] if a > 0 else s0
      prev = [x for x in self.moves_along if x < s1 - 1.0]
      if prev:  # nor back across the move before (its lanes may not lead on), unless nav's own change is
        lower = max(lower, min(prev[-1] + CROSSING_CLEAR, s0))
      upper = keys[b + 1][0] if b + 1 < len(keys) and abs(keys[b + 1][1] - keys[b][1]) < EPS else s1
      shortest = LC_SHORTEST * max(lanes, 1.0)
      s_new = min(max(at_time(s1, -want), lower), along_of(path_s(s1) - shortest))
      k = segment_of(route, s0)
      fwy = (route.classes[k] if k < len(route.classes) else "") in FREEWAY
      if not fwy and any(s_new - CROSSING_CLEAR < c < s1 for c in crossings) and not any(s0 - CROSSING_CLEAR < c < s1 for c in crossings):
        s_new = max(c for c in crossings if s_new - CROSSING_CLEAR < c < s1) + CROSSING_CLEAR
      e_new = s1
      if secs(s_new, s1) < (1 - LC_FIT) * want and s0 <= keys[0][0] + 1.0:
        e_new = max(s1, min(at_time(s_new, want), upper - 1.0))
      if abs(secs(s_new, e_new) - want) >= abs(secs(s0, s1) - want) or e_new - s_new < shortest - 1e-6:
        continue  # no better than it was
      first = a
      while first > 0 and keys[first - 1][0] > s_new + EPS:  # holds it now starts before
        first -= 1
      after = b + 1
      while after < len(keys) and keys[after][0] < e_new - EPS:  # holds it now ends after
        after += 1
      keys[first:after] = self._replace_chain(keys, a, b, s_new, e_new)
    if any(k2[0] < k1[0] - EPS for k1, k2 in zip(keys, keys[1:], strict=False)):
      return original  # a re-placing gone wrong: as it was
    return keys

  def off_jogs(self, route, keys: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The keys with each single lane change moved off the route's jogs (route_jogs), where the lane line already
    swerves: wholly before the jog where there's room back to the key before, else after it where there's room on to
    the key after, neither onto a stop line it was clear of; one with neither stays, logged as a jog_change anomaly."""
    spans = route_jogs(route)
    if not spans:
      return keys
    keys = list(keys)
    for a, b in reversed(chains(keys)):
      if b != a + 1:
        continue
      (s0, la), (s1, lb) = keys[a], keys[b]
      hit = next(((j0, j1) for j0, j1 in spans if j0 < s1 and s0 < j1), None)
      if hit is None:
        continue
      length = s1 - s0
      lower = keys[a - 1][0] if a > 0 else s0
      upper = keys[b + 1][0] if b + 1 < len(keys) else s1

      def clear(x0, x1, crossings=()):
        return not any(j0 < x1 and x0 < j1 for j0, j1 in spans) and not any(x0 - CROSSING_CLEAR < c < x1 for c in crossings)
      # before or after it; first where it crosses no stop line or junction node, as nav keeps changes off them
      options = [(hit[0] - length, hit[0]), (hit[1], hit[1] + length)]
      # never onto a stop line it was clear of: on the jog is better than across the line into the junction
      stops = () if any(s0 - CROSSING_CLEAR < c < s1 for c in route.stops) else route.stops
      ok = [o for o in options if o[0] >= lower and o[1] <= upper - 1.0 and clear(*o, stops)]
      best = [o for o in ok if clear(*o, sorted(route.stops + route.junctions))] or ok
      if best:
        keys[a], keys[b] = (best[0][0], la), (best[0][1], lb)
      else:
        self.jog_changes.append((s0, s1))
    return keys

  @staticmethod
  def _chain_lanes(keys, a: int, b: int) -> float:
    """Lanes a change crosses: its ramps' changes, without its renumbering steps."""
    return sum(keys[j + 1][1] - keys[j][1] for j in range(a, b) if keys[j + 1][0] - keys[j][0] > EPS)

  @staticmethod
  def _retarget_chain(keys, a: int, b: int, lane: float) -> list[tuple[float, float]]:
    """A change's keys (a to b) ending in `lane` instead, at the same places: its ramps scaled, its renumbering steps
    kept."""
    moved = MapDrive._chain_lanes(keys, a, b)
    scale = (moved - (keys[b][1] - lane)) / moved
    out = [keys[a]]
    for j in range(a, b):
      (s, l0), (s1, l1) = keys[j], keys[j + 1]
      out.append((s1, out[-1][1] + ((l1 - l0) if s1 - s <= EPS else (l1 - l0) * scale)))
    out[-1] = (out[-1][0], lane)
    return out

  @staticmethod
  def _replace_chain(keys, a: int, b: int, s0: float, s1: float) -> list[tuple[float, float]]:
    """A change's keys (a to b) moved to run from s0 to s1, linearly across (shaped() makes it a quintic): its lanes
    moved, in the end's numbering, at the new share for each place, and its renumbering steps (two keys at one place)
    where they were, each lane before one in the numbering before it."""
    steps = [(keys[j][0], keys[j + 1][1] - keys[j][1]) for j in range(a, b)
             if keys[j + 1][0] - keys[j][0] <= EPS and abs(keys[j + 1][1] - keys[j][1]) > EPS]
    moved = sum(keys[j + 1][1] - keys[j][1] for j in range(a, b) if keys[j + 1][0] - keys[j][0] > EPS)
    end = keys[b][1]

    def lane(s: float, before: bool = False) -> float:
      """In the numbering at s (before a step there, with `before`)."""
      f = min(max((s - s0) / max(s1 - s0, EPS), 0.0), 1.0)
      later = sum(jump for pos, jump in steps if pos > s + EPS or (before and abs(pos - s) <= EPS))
      return end - moved * (1.0 - f) - later
    out = [(s0, lane(s0, before=True))]
    for pos, _ in steps:
      out += [(pos, lane(pos, before=True)), (pos, lane(pos))]
    out.append((s1, lane(s1)))
    return sorted(out, key=lambda k: k[0])

  def _unclear(self, along: float) -> bool:
    """Whether a place (m along the route) is near a junction node or a sharp bend of the route's line."""
    u = self.unclear
    if not len(u):
      return False
    i = int(np.searchsorted(u, along))
    return min(abs(u[min(i, len(u) - 1)] - along), abs(u[max(i - 1, 0)] - along)) < JUNCTION_CLEAR

  @staticmethod
  def _offset_from(line: np.ndarray, p: np.ndarray) -> float:
    """m right of a polyline's start (its first 30 m) that p is."""
    n = min(len(line), 31)
    a, b = line[:n - 1], line[1:n]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9), 0.0, 1.0)
    d = np.hypot(*(a + ab * t[:, None] - p).T)
    i = int(np.argmin(d))
    seg = max(float(np.hypot(*ab[i])), 1e-6)
    return float((p[0] - a[i, 0]) * ab[i, 1] - (p[1] - a[i, 1]) * ab[i, 0]) / seg

  def _derive(self, route, intent: np.ndarray, off: np.ndarray, t: float, start=None):
    """The driven path (intent moved `off` m right; from the car's pose `start` (pos, heading, speed) along a cubic
    where it heads another way) and what's read along it: curvature, the route's along and right, the stops and
    turns, the speed profile."""
    th = headings(intent)
    target = intent + np.stack([np.sin(th), -np.cos(th)], axis=1) * off[:, None]
    if start is not None:
      target = hermite_join(target, *start)
    self.intent, self.path = intent, target
    self.s_path = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(target, axis=0).T))))
    self.theta = headings(target)
    k_raw = np.gradient(self.theta, self.s_path)
    self.kappa_path = smooth(k_raw, max(int(CURV_SMOOTH / STEP) | 1, 1))
    self.route_s, self.route_r = along_route(route, target, route.at - 5.0)
    self.route = route
    self.path0, self.nudge_off, self.nudge_fixed = target, np.zeros(len(target)), np.zeros(len(target))
    self.theta0 = self.theta
    self.nudge_groups, self.nudge_blocked = [], []
    self.room_lo, self.room_hi = self._nudge_room()
    self.borrow_lo, self.borrow_hi = np.full(len(target), np.nan), np.full(len(target), np.nan)
    self.room_tried = np.zeros(len(target), bool)
    self._turn_alongs = [tr.dist for tr in (getattr(route, "_turns", None) or []) if abs(tr.angle) >= CLAMP_TURN] + \
                        [sc for sc, turned in route.lanes.corners if abs(turned) >= CLAMP_TURN]
    # where the route's own line says little about the road beside it: junction nodes, and its points where it bends
    # sharply (a lane line or the car cuts inside such a corner, past the segment's own kerbs)
    d = np.diff(route.points[:, :2], axis=0)
    h = np.arctan2(d[:, 1], d[:, 0])
    turn = np.degrees(np.abs((np.diff(h) + np.pi) % (2 * np.pi) - np.pi))
    self.unclear = np.sort(np.concatenate((np.asarray(route.junctions, float), route.along[1:-1][turn > BEND_DEG])))
    self.s = 0.0
    # path m from before a splice mean nothing on the new path: the stall clock runs on from 0 m, a crossing's hold goes
    self.progress, self.cross_hold = (0.0, self.progress[1]), None
    # stops ahead, as path m
    self.stops = []
    for along, kind in zip(route.stops, route.stop_kinds, strict=False):
      if along <= route.at + 1.0:
        continue
      s_line = float(np.interp(along, self.route_s, self.s_path))
      self.stops.append({"along": float(along), "kind": kind, "s": s_line, "done": False, "t_stop": None,
                         "target": s_line - self.front - self.style["stop_margin"]})
    arrive_along = route.length - float(self.c["stop_before"])
    self.s_arrive = float(np.interp(max(arrive_along, route.at + 5.0), self.route_s, self.s_path))
    # the static speed profile: the road's speed, the curvature, nav's turn speeds, the arrival
    k = np.clip(np.searchsorted(route.along, self.route_s, side="right") - 1, 0, max(len(route.points) - 2, 0))
    lim = np.array([road_speed(route, int(j)) for j in k]) * self.style["speed"]
    if self.c.get("speed"):
      lim = np.minimum(lim, float(self.c["speed"]))
    # by the curvature's envelope (its largest within CURV_SMOOTH either way): the average smooths a short S (a jog, a
    # lane change across a bend) down to less than the car must take
    n = max(int(CURV_SMOOTH / STEP), 1)
    k_abs = np.abs(smooth(k_raw, 3))
    env = k_abs.copy()
    for j in range(1, n + 1):
      env[j:] = np.maximum(env[j:], k_abs[:-j])
      env[:-j] = np.maximum(env[:-j], k_abs[j:])
    curve = np.sqrt(self.style["a_lat"] / np.maximum(np.maximum(env, np.abs(self.kappa_path)), 1e-4))
    # no faster than the lateral jerk allows where the curvature changes (jerk = v^3 dkappa/ds)
    dk = np.abs(np.gradient(self.kappa_path, self.s_path))
    curve = np.minimum(curve, np.cbrt(JERK_SHARE * self.style["j_lat"] / np.maximum(dk, 1e-6)))
    v = np.minimum(lim, curve)
    why = np.where(curve < lim, REASONS.index("curve"), REASONS.index("limit"))
    for d, side, exit_h, angle in route.turns(route.length):
      s_t = float(np.interp(route.at + d, self.route_s, self.s_path))
      cap = Turn(d, side, exit_h, abs(angle)).speed(self.tune) * self.style["turn"]
      m = (np.abs(self.s_path - s_t) < TURN_SPAN) & (v > cap)
      v[m], why[m] = cap, REASONS.index("turn")
    end = self.s_path >= self.s_arrive
    v[end], why[end] = 0.0, REASONS.index("arrive")
    b = self.style["decel"] * 0.8
    for i in range(len(v) - 2, -1, -1):
      ds = self.s_path[i + 1] - self.s_path[i]
      lim_b = math.sqrt(v[i + 1] ** 2 + 2 * b * ds)
      if lim_b < v[i]:
        v[i], why[i] = lim_b, why[i + 1]
    self.v_static, self.why_static = v, why
    self.mans = maneuvers(route)
    self.corners = self._corners(route)
    self.done_mans: set[int] = set()

  def _changes(self, keys):
    out = []
    for a, b in chains(keys):
      s0 = float(np.interp(keys[a][0], self.route_s, self.s_path))
      s1 = float(np.interp(keys[b][0], self.route_s, self.s_path))
      d = keys[b][1] - keys[a][1]
      # the lanes' numbering may step inside it: the side is the way across the road, by the path's right offset
      r0 = float(np.interp(s0, self.s_path, self.route_r))
      r1 = float(np.interp(s1, self.s_path, self.route_r))
      side = "right" if (r1 - r0 if abs(r1 - r0) > 1.0 else d) > 0 else "left"
      out.append((s0, s1, side, round(r1 - r0, 2)))
    return out

  def _plan_anomalies(self, route, t: float):
    """Map anomalies the plan itself shows: the car's side over a kerb along it, GTA's lanes elsewhere than the map's,
    a stop line far from its junction."""
    lanes = route.lanes
    junctions = np.asarray(route.junctions, float)
    found: dict[tuple, list] = {}  # (kind, why): [(path index, details)], each run of them one anomaly
    every = 5
    for i in range(0, len(self.path), every):
      rs, rr = self.route_s[i], self.route_r[i]
      if self._unclear(rs) or self.s_path[i] < self.join_m:
        continue
      k = segment_of(route, rs)
      sec = lanes.section_at(rs, k)
      if sec is None or not sec.lanes:
        continue
      lo, hi = sec.edges
      if rr - self.half_width < lo - KERB_MARGIN or rr + self.half_width > hi + KERB_MARGIN:
        found.setdefault(("kerb_contact", ""), []).append((i, {"right": round(float(rr), 2), "edges": [round(lo, 2), round(hi, 2)]}))
      link = route.links[k] if k < len(route.links) else None
      if link is not None and link.lanes:
        lane = lane_at(self.keys, rs)
        if abs(lane - round(lane)) > 0.2:
          continue
        back = len([sp for sp in sec.spans if sp.heading == -1])
        if link.lanes != sec.lanes or link.back != back:
          found.setdefault(("gta_link_offset", "lane counts"), []).append((i, {"map": [sec.lanes, back], "gta": [link.lanes, link.back]}))
          continue
        gta_lo, gta_hi = link.inner - link.back * link.width, link.inner + link.lanes * link.width
        room = min(rr - gta_lo, gta_hi - rr) - self.half_width
        # the map's lanes wider than GTA's on the near side (a lane placed out towards GTA's barrier or kerb; a shoulder
        # beyond them is no lane)
        lanes_lo, lanes_hi = sec.spans[0].left, sec.spans[-1].right
        wider = (lanes_lo < gta_lo - EDGE_WIDER) if rr - gta_lo < gta_hi - rr else (lanes_hi > gta_hi + EDGE_WIDER)
        if room < EDGE_CLEAR and wider:
          found.setdefault(("gta_edge", ""), []).append((i, {"room": round(float(room), 2), "right": round(float(rr), 2),
                                                            "gta_edges": [round(gta_lo, 2), round(gta_hi, 2)],
                                                            "map_edges": [round(float(lo), 2), round(float(hi), 2)]}))
        gta = link.inner + (round(lane) + 0.5) * link.width
        mine = float(sec.offset(round(lane)))
        if abs(gta - mine) > GTA_OFFSET:
          found.setdefault(("gta_link_offset", "lane centre"), []).append((i, {"lane": int(round(lane)), "map": round(mine, 2), "gta": round(gta, 2)}))
    for (kind, why), hits in found.items():
      runs = [[hits[0]]]
      for h in hits[1:]:
        if h[0] - runs[-1][-1][0] <= 3 * every:
          runs[-1].append(h)
        else:
          runs.append([h])
      for run in runs:
        i, detail = run[len(run) // 2]
        extra = {"why": why} if why else {}
        self._anomaly(kind, t, self.path[i], float(self.s_path[i]), planned=True, length=round(float(self.s_path[run[-1][0]] - self.s_path[run[0][0]]) + every, 1),
                      **extra, **detail)
    for s0, s1 in self.jog_changes:
      self._anomaly("jog_change", t, route_point(route, s0), float(np.interp(s0, self.route_s, self.s_path)),
                    span=[round(s0, 1), round(s1, 1)])
    for c in self.corners:  # the plan cutting a junction's corner kerb
      near = np.flatnonzero(np.abs(np.asarray(self.route_s[:len(self.path)]) - c["along"]) < CORNER_SPAN)
      depth = self._corner_hits(c, self.path[near], np.asarray(self.theta, float)[near], self.body)
      hits = near[np.nan_to_num(depth) > KERB_MARGIN]
      if len(hits):
        i = int(hits[len(hits) // 2])
        self._anomaly("kerb_contact", t, self.path[i], float(self.s_path[i]), planned=True, corner=c["side"],
                      depth=round(float(np.nanmax(depth)), 2),
                      corner_at=[round(float(v), 1) for v in c["corner"]])
    islands = getattr(lanes, "islands", None) or []
    if islands and len(self.path) >= 2:
      push, _ = self._island_push(self.path, self.theta, islands, cap=None)
      hits = np.flatnonzero((np.abs(push) > KERB_MARGIN) & (self.s_path >= self.join_m))
      if len(hits):
        i = int(hits[np.argmax(np.abs(push[hits]))])
        self._anomaly("island_contact", t, self.path[i], float(self.s_path[i]), planned=True, depth=round(float(abs(push[i])), 2))
    for st in self.stops:
      after = junctions[junctions > st["along"] - 1.0] if len(junctions) else junctions
      gap = float(after[0] - st["along"]) if len(after) else math.inf
      if gap > STOP_LINE_FAR:
        self._anomaly("stop_line_far", t, route_point(route, st["along"]), st["s"], kind_line=st["kind"],
                      to_junction=None if math.isinf(gap) else round(gap, 1))

  def _corners(self, route, line: np.ndarray | None = None, line_s: np.ndarray | None = None,
               bends: list | None = None) -> list[dict]:
    """Each turn's corner kerb on its inside (the route's turns, and its corners where the lane line takes a fillet): the
    kerb lines of the road in and the road out (the map's cross-sections a leg from the turn's point, CORNER_LEG or half
    the way to the next corner, carried on straight), meeting at the corner: a median's edge on the far side, else the
    road's (_kerb). Only where that's a junction's corner the line (the path, or `line` at line_s m along the route)
    clears: a turn of CORNER_ANGLE at a junction node, the corner near the turn's point, and the line inside both kerb
    lines on the legs. Turns whose kerb runs inside the block (the road bends there), with CORNER_LEGS_SHORT's legs too,
    are added to `bends`."""
    line = self.path if line is None else line
    line_s = self.route_s if line_s is None else line_s
    out = []
    lanes = route.lanes
    junctions = np.asarray(route.junctions, float)
    route.turns(0.0)
    sites = [(tr.dist, tr.side, abs(tr.angle)) for tr in getattr(route, "_turns", None) or []]
    sites += [(sc, "left" if turned > 0 else "right", abs(turned)) for sc, turned in lanes.corners
              if not any(abs(sc - s0) < CORNER_SAME for s0, _, _ in sites)]
    sites.sort()
    for n, (dist, side, angle) in enumerate(sites):
      room = min([abs(dist - s0) for k, (s0, _, _) in enumerate(sites) if k != n] + [2 * CORNER_LEG])
      leg = min(CORNER_LEG, room / 2)
      a, b = dist - leg, dist + leg
      if leg < CORNER_LEG_MIN or not CORNER_ANGLE[0] <= angle <= CORNER_ANGLE[1] or a < 10.0 or b > route.length - 10.0:
        continue
      if not len(junctions) or np.abs(junctions - dist).min() > CORNER_NODE:
        continue
      # the legs shortened where the road beyond them bends away into the block (a side road curving off)
      tries = [self._corner_at(route, dist, side, leg)]
      if tries[0] == "bend":
        tries += [self._corner_at(route, dist, side, x) for x in CORNER_LEGS_SHORT if CORNER_LEG_MIN <= x < leg]
      c = next((c for c in tries if isinstance(c, dict)), None)
      if c is not None:
        ia, ib = (int(np.argmin(np.abs(line_s - x))) for x in (dist - c["leg"], dist + c["leg"]))
        if (line[ia] - c["ka"]) @ c["na"] > -CORNER_INSIDE or (line[ib] - c["kb"]) @ c["nb"] > -CORNER_INSIDE:
          c = None  # the line outside a leg's kerb line: the line isn't the kerb
      if c is not None:
        out.append(c)
      elif tries[0] == "bend" and bends is not None:
        bends.append(float(dist))  # the map's own kerb runs inside the block: a bend, a slip or a wide corner
    return out

  def _corner_at(self, route, dist: float, side: str, leg: float) -> dict | str | None:
    """A turn's corner kerb block from its legs' cross-sections `leg` m either side of its point (_corners): "bend"
    where the map's kerb between runs into it, None where there's none."""
    a, b = dist - leg, dist + leg
    lanes = route.lanes
    sa, sb = lanes.section_at(a), lanes.section_at(b)
    if sa is None or sb is None or not sa.lanes or not sb.lanes:
      return None
    kerb_a, kerb_b = self._kerb(sa, side), self._kerb(sb, side)
    chord = min(10.0, leg)
    pa, pb = route_point(route, a), route_point(route, b)
    da, db = pa - route_point(route, a - chord), route_point(route, b + chord) - pb
    da, db = da / max(np.hypot(*da), 1e-6), db / max(np.hypot(*db), 1e-6)
    right = side == "right"
    # outward of the road, into the corner block
    na = np.array([da[1], -da[0]]) * (1 if right else -1)
    nb = np.array([db[1], -db[0]]) * (1 if right else -1)
    ka = pa + np.array([da[1], -da[0]]) * kerb_a[0]
    kb = pb + np.array([db[1], -db[0]]) * kerb_b[0]
    m = np.array([da, -db]).T
    if abs(np.linalg.det(m)) < 0.2:
      return None  # the legs near parallel: no corner to speak of
    u = np.linalg.solve(m, kb - ka)
    corner = ka + da * u[0]
    if np.hypot(*(corner - route_point(route, dist))) > CORNER_NEAR:
      return None
    if self._road_in_block(route, a, b, ka, na, kb, nb, side):
      return "bend"
    return {"along": float(dist), "side": side, "ka": ka, "na": na, "kb": kb, "nb": nb, "corner": corner,
            "radius": min(kerb_a[1], kerb_b[1]), "leg": leg}

  def _kerb(self, sec, side: str) -> tuple[float, float]:
    """m right of the line of a cross-section's kerb on `side`, and how rounded its corner is taken: the road's edge on
    the near side, KERB_RADIUS; on the far side our own carriageway's edge where MEDIAN_GAP lies between the directions,
    MEDIAN_RADIUS; else the road's far edge (nothing coming the other way, or crossing the oncoming lanes is the turn,
    cutting past the road's edge isn't), FAR_RADIUS."""
    near = side == ("right" if self.c["drive_on_right"] else "left")
    if near:
      return (sec.edges[1] if side == "right" else sec.edges[0]), KERB_RADIUS
    if not any(sp.heading == -1 for sp in sec.spans):
      return (sec.edges[1] if side == "right" else sec.edges[0]), FAR_RADIUS
    ours = sec.ours
    if side == "left" and sec.first > 0 and ours[0].left - sec.spans[sec.first - 1].right >= MEDIAN_GAP:
      return ours[0].left, MEDIAN_RADIUS
    beyond = sec.first + sec.lanes
    if side == "right" and beyond < len(sec.spans) and sec.spans[beyond].left - ours[-1].right >= MEDIAN_GAP:
      return ours[-1].right, MEDIAN_RADIUS
    return (sec.edges[1] if side == "right" else sec.edges[0]), FAR_RADIUS

  def _road_in_block(self, route, a: float, b: float, ka, na, kb, nb, side: str) -> bool:
    """Whether the map's kerb on `side` between a and b m along (the route's cross-sections there, every 2 m) reaches
    more than 1 m into the corner block the two kerb lines make."""
    lanes = route.lanes
    for s_ in np.arange(a, b, 2.0):
      k = segment_of(route, float(s_))
      sec = lanes.section_at(float(s_), k)
      if sec is None or not sec.lanes:
        continue
      kerb = self._kerb(sec, side)
      if kerb is None:
        continue
      p0, p1 = route.points[k], route.points[k + 1]
      d = (p1 - p0) / max(float(np.hypot(*(p1 - p0))), 1e-6)
      e = route_point(route, float(s_)) + np.array([d[1], -d[0]]) * kerb[0]
      if min(float((e - ka) @ na), float((e - kb) @ nb)) > 1.0:
        return True
    return False

  @staticmethod
  def _corner_hit(c: dict, pos, theta: float, body: tuple[float, float, float] = (FRONT, FRONT, HALF_WIDTH)) -> float | None:
    """How deep (m) the car (its origin at pos, heading theta radians from east; body: m to its front, rear and side)
    reaches into a corner's kerb block, the block's corner rounded by its radius; None where it doesn't."""
    d = MapDrive._corner_hits(c, np.asarray(pos, float)[None, :2], np.array([theta], float), body)[0]
    return None if np.isnan(d) else float(d)

  @staticmethod
  def _corner_hits(c: dict, pos: np.ndarray, theta: np.ndarray, body: tuple[float, float, float]) -> np.ndarray:
    """_corner_hit at each of n places (pos (n, 2), theta (n,)), nan where it doesn't reach in: by points round the
    body's outline BODY_STEP apart, as the block's corner can stand beside its flank between its corners."""
    out = _outline(*body)
    f = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    lat = np.stack([-f[:, 1], f[:, 0]], axis=1)
    q = pos[:, None, :] + f[:, None, :] * out[None, :, :1] + lat[:, None, :] * out[None, :, 1:]
    d1, d2 = (q - c["ka"]) @ c["na"], (q - c["kb"]) @ c["nb"]
    r = c.get("radius", KERB_RADIUS)
    depth = np.where((d1 < r) & (d2 < r), r - np.hypot(r - d1, r - d2), np.minimum(d1, d2))  # into the rounded corner
    depth = np.where((d1 > 0) & (d2 > 0) & (depth > 0), depth, -np.inf).max(axis=1)
    return np.where(np.isfinite(depth), depth, np.nan)

  # *** the plan kept on its lane ***

  def _clamp(self, route, keys: list[tuple[float, float]], intent: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The intent moved back where it strays (the lane line's fillets and eased corners cut across what's beside the
    lane): our body out of each turn's inside corner kerb (_corners) by CLAMP_MARGIN; outside lane changes and the
    corners of turns of CLAMP_TURN or more (but bends), our body within its lane, CLAMP_MARGIN in from its edges, and
    near a fork within FORK_ROOM of the lane's centre on the gore's side; our body CLAMP_MARGIN clear of the raised
    islands (_island_push, up to ISLAND_MAX), the lane's own limits let go beside one. Each push eased in and out over
    CLAMP_EASE m, at most CLAMP_MAX. Also how far (m) the line may then move left and right in its lane (for the in-lane bias and
    wander): (intent, left, right). The pushes are kept in `clamps`, plan_clamp anomalies."""
    self.clamps = []
    along, _ = along_route(route, intent, route.at - 5.0)
    bends: list[float] = []
    self._corners(route, intent, along, bends)
    lanes_ = self._lane_room(route, keys, np.arange(along[0], along[-1] + LANE_GRID, LANE_GRID), bends)
    self.lane_room = lanes_
    for n in range(4):  # a push turns the body, and moves it nearer another corner
      need, why, room_l, room_r = self._strays(route, intent, lanes_)
      if n == 3 or not np.any(need):
        break
      for a, b in _runs(np.abs(need) > 0.05):
        i = a + int(np.argmax(np.abs(need[a:b])))
        self.clamps.append({"i": i, "push": round(float(need[i]), 2), "why": why[i], "length": float(b - a) * STEP})
      th = headings(intent)
      push = eased(need, int(CLAMP_EASE / STEP))
      intent, _ = resample(intent + np.stack([np.sin(th), -np.cos(th)], axis=1) * push[:, None])
    return intent, room_l, room_r

  def _lane_room(self, route, keys, along: np.ndarray, bends: list[float]):
    """At each place along the route (along, every LANE_GRID m): our lane's centre (m right of the route's line; NaN
    where the lane is no guide: lane changes, jogs, tapers, and about turns of CLAMP_TURN or more but `bends`), how far
    left and right of it our middle may be (in by our half width and CLAMP_MARGIN; near a fork, at most FORK_ROOM
    towards its gore), and why."""
    n = len(along)
    centre, left, right, width = np.full(n, np.nan), np.zeros(n), np.zeros(n), np.full(n, np.nan)
    why = ["lane"] * n
    lanes = route.lanes
    turns = [tr.dist for tr in (getattr(route, "_turns", None) or []) if abs(tr.angle) >= CLAMP_TURN]
    turns += [sc for sc, turned in lanes.corners if abs(turned) >= CLAMP_TURN]
    turns = np.asarray([x for x in turns if not any(abs(x - y) < CORNER_SAME for y in bends)], float)
    jogs = route_jogs(route)
    forks = [(f.along, f.side) for f in route.forks if f.keep]
    ks = [k[0] for k in keys]
    room0 = self.half_width + CLAMP_MARGIN
    for i in range(n):
      rs = float(along[i])
      j = bisect.bisect_right(ks, rs) - 1
      lane = lane_at(keys[max(j, 0):j + 2], rs) if 0 <= j < len(keys) - 1 else keys[min(max(j, 0), len(keys) - 1)][1]
      if abs(lane - round(lane)) > 0.02:
        continue  # a lane change under way
      if len(turns) and np.abs(turns - rs).min() < CORNER_ZONE or any(j0 < rs < j1 for j0, j1 in jogs):
        continue
      sec = lanes.section_at(rs, segment_of(route, rs))
      if sec is None or not sec.lanes or not sec.lo <= round(lane) <= sec.hi:
        continue
      sp = sec.spans[sec.first + int(round(lane))]
      width[i] = sp.right - sp.left
      room = width[i] / 2 - room0
      if room < 0.1:
        continue  # narrower than the car: a lane closing or opening, which the lane line moves across
      centre[i], left[i], right[i] = sp.centre, room, room
      gore = next((side for fa, side in forks if -FORK_BEFORE < rs - fa < FORK_AFTER), None)
      if gore == "right" and room > FORK_ROOM:  # the branch taken is the right one: the gore on our left
        left[i], why[i] = FORK_ROOM, "fork"
      elif gore == "left" and room > FORK_ROOM:
        right[i], why[i] = FORK_ROOM, "fork"
    # where the lane's centre jumps (a jog the lane line eases across), or it runs out or in (a taper to narrower than
    # the car), it's no guide
    w = int(CORNER_ZONE / LANE_GRID)
    for j in np.flatnonzero((np.abs(np.diff(centre)) > 1.0) | (width[1:] < 2 * room0 + 0.2)):
      centre[max(j - w, 0):j + w] = np.nan
    # but across a stretch without a guide between two with one, in the same lane with no change, step, turn or jog
    # between (a junction's own sections, which say little of where our lane runs through it: seed 43 followed them
    # onto the centre line), the lane carried across along the route's line, the lesser room
    for a, b in _runs(np.isnan(centre)):
      if a == 0 or b >= n or along[b] - along[a - 1] > CARRY_MAX:
        continue
      s0, s1 = float(along[a - 1]), float(along[b])
      lane0 = lane_at(keys, s0)
      if any(abs(lane_at(keys, float(x)) - lane0) > 0.02 for x in along[a - 1:b + 1]) or any(j0 < s1 and s0 < j1 for j0, j1 in jogs) or \
         (len(turns) and ((turns > s0 - CORNER_ZONE) & (turns < s1 + CORNER_ZONE)).any()):
        continue
      f = (along[a:b] - s0) / (s1 - s0)
      centre[a:b] = centre[a - 1] + (centre[b] - centre[a - 1]) * f
      left[a:b], right[a:b] = min(left[a - 1], left[b]), min(right[a - 1], right[b])
      why[a:b] = ["lane"] * (b - a)
    return along, centre, left, right, why

  def _strays(self, route, line: np.ndarray, lanes_):
    """How far (m, right positive) each point of a line must move to keep our body where _clamp keeps it (lanes_: its
    _lane_room), why, and the room it has left and right in its lane after."""
    along, right = along_route(route, line, route.at - 5.0)
    grid, centre, left, right_, kind = lanes_
    g = np.clip(np.round((along - grid[0]) / LANE_GRID).astype(int), 0, len(grid) - 1)
    off = right - centre[g]
    ok = ~np.isnan(off)
    lim_l, lim_r = left[g], right_[g]
    # the moves right that keep it clear: at least lo, at most hi
    lo = np.where(ok & (off < -lim_l), np.minimum(-lim_l - np.nan_to_num(off), CLAMP_MAX), -np.inf)
    hi = np.where(ok & (off > lim_r), np.maximum(lim_r - np.nan_to_num(off), -CLAMP_MAX), np.inf)
    why = [kind[j] for j in g]
    room_l = np.where(ok, np.clip(lim_l + np.nan_to_num(off), 0.0, 0.45), 0.45)
    room_r = np.where(ok, np.clip(lim_r - np.nan_to_num(off), 0.0, 0.45), 0.45)
    th = headings(line)
    front, rear, hw = self.body
    m = CLAMP_MARGIN
    islands = getattr(route.lanes, "islands", None) or []
    if islands:
      push, near = self._island_push(line, th, islands)
      if near.any():  # by an island the lane is no guide: the island's push holds the plan off it, the lane's doesn't pull it back
        w = int(CLAMP_EASE / STEP)
        zone = np.convolve(near.astype(float), np.ones(2 * w + 1), mode="same") > 0.5
        lo[zone], hi[zone] = -np.inf, np.inf
        room_l[zone], room_r[zone] = 0.0, 0.0
        for i in np.flatnonzero(np.abs(push) > 0.05):
          if push[i] > 0:
            lo[i] = min(float(push[i]), ISLAND_MAX)
          else:
            hi[i] = max(float(push[i]), -ISLAND_MAX)
          why[i] = "island"
    for c in self._corners(route, line, along):
      # within its legs: beyond them the road may bend away from their kerb lines
      near = np.flatnonzero(np.abs(along - c["along"]) < c["leg"])
      depth = np.minimum(np.nan_to_num(self._corner_hits(c, line[near], th[near], (front + m, rear + m, hw + m))), CLAMP_MAX)
      for i, d in zip(near[depth > 0.05], depth[depth > 0.05], strict=True):
        if c["side"] == "right":
          hi[i] = min(hi[i], -d)
        else:
          lo[i] = max(lo[i], d)
        why[i] = "corner"
    both = (lo > 0) & (hi < 0)  # pushed both ways: halfway
    need = np.where(both, (np.where(both, lo, 0.0) + np.where(both, hi, 0.0)) / 2, np.where(lo > 0, lo, np.where(hi < 0, hi, 0.0)))
    need[: int(10.0 / STEP)] = 0.0  # where the car joins its line
    return need, why, room_l, room_r

  def _island_push(self, line: np.ndarray, th: np.ndarray, islands: list[np.ndarray],
                   cap: float | None = ISLAND_MAX) -> tuple[np.ndarray, np.ndarray]:
    """For each point of a line (headings th): how far (m, right positive) it must move for our body, CLAMP_MARGIN
    round it, to clear the raised islands (outlines [K, 2]), the shorter way round each, 0 where it's clear; and whether
    an island is within ISLAND_ROOM beside that body. Each island is passed on one side all along (the side it takes
    less pushing to), so a line through its middle isn't pushed both ways in turn. An island needing more than `cap` is
    left out (None: none is)."""
    from openpilot.tools.sim.bridge.gta5.map.junctions import densify
    need, near = np.zeros(len(line)), np.zeros(len(line), bool)
    front, rear, hw = self.body
    m = CLAMP_MARGIN
    half = hw + m
    for ring in islands:
      pts = densify(np.vstack([ring, ring[:1]]), 0.25)
      c = ring.mean(0)
      reach = float(np.hypot(*(ring - c).T).max()) + max(front, rear) + half + ISLAND_ROOM
      hits, close = [], []  # (point, push right that clears it, push left that does); points it's beside
      for i in np.flatnonzero(np.hypot(*(line - c).T) < reach):
        f = np.array([math.cos(th[i]), math.sin(th[i])])
        rel = pts - line[i]
        x, y = rel @ f, rel @ np.array([-f[1], f[0]])  # along, left of the line
        ys = y[(x >= -(rear + m)) & (x <= front + m)]
        if not len(ys) or ys.min() >= half + ISLAND_ROOM or ys.max() <= -(half + ISLAND_ROOM):
          continue
        close.append(i)
        if ys.min() < half and ys.max() > -half:
          hits.append((i, half - float(ys.min()), half + float(ys.max())))
      go_right = sum(h[1] for h in hits) <= sum(h[2] for h in hits)
      if cap is not None and hits and max(h[1] if go_right else h[2] for h in hits) > cap:
        continue  # through its middle: no push mends that
      near[close] = True
      if hits:
        for i, right, left in hits:
          d = right if go_right else -left
          if abs(d) > abs(need[i]):
            need[i] = d
    return need, near

  def _splice(self, route, state: dict, t: float, shift: int = 0):
    """A new route under way: the old path kept past the lane change under way and 3 s on, or where the new route runs
    on along the old one's roads (a cruise route walked on, a reroute that differs only far ahead) to SPLICE_BACK m short
    of where it leaves them, past any lane change there; then the new plan's, the gap to it closed as a lane change
    would. With `shift`, the same route planned again with the lane change the player asked for (_make_plan), the old
    path kept only 0.5 s on, as its signal comes on at once; for an arrow key's turn (route.prompt) 3 s on but leaving
    room before the turn (PROMPT_ROOM_S), down to PROMPT_KEEP_S s."""
    old_path, old_s, s, v = self.path, self.s_path, self.s, float(state.get("vEgo") or 0.0)
    if shift:
      cut = s + max(0.5 * v, 3.0)
    elif getattr(route, "prompt", False):
      room = max(PROMPT_ROOM_S * v, PROMPT_ROOM)  # left before where the turn leaves the old roads
      cut = max(s + max(PROMPT_KEEP_S * v, 3.0), min(s + max(3.0 * v, 10.0), self._same_roads(route) - room))
    else:
      cut = max(s + max(3.0 * v, 10.0), self._same_roads(route) - SPLICE_BACK)
    for _ in range(2):  # a lane change past the cut can end where another begins
      for s0, s1, *_ in self.changes:
        if s0 <= s <= s1 or s0 <= cut <= s1:
          cut = max(cut, s1)
    for g in self.nudge_groups:  # a shift begun kept whole: the new path is from the plan, without it
      if g["start"] <= cut and g["end"] >= s:
        cut = max(cut, g["end"])
    part = (old_s >= s - 2.0) & (old_s <= cut)  # from the car on: the new route starts there
    keep, old_intent = old_path[part], self.intent[part]
    if not self._make_plan(route, state, t, shift):
      return
    if len(keep) >= 2:
      j, d = project(self.path, self.s_path, keep[-1], 0.0, window=self.s_path[-1])
      after = self.s_path > j + 2.0
      tail, tail_i = self.path[after], self.intent[after]
      if d > SPLICE_GAP and len(tail) >= 2:
        # the kept path ends beside the new one (another lane): over to it as a lane change, not a step
        th = headings(tail)
        right = np.stack([np.sin(th), -np.cos(th)], axis=1)
        at_j = np.array([np.interp(j, self.s_path, self.path[:, 0]), np.interp(j, self.s_path, self.path[:, 1])])
        gap = float((keep[-1] - at_j) @ right[0])
        over = max(MIN_LC_M, lc_seconds(self.style, gap / LANE_W) * max(v, 5.0))
        shift = gap * (1.0 - quintic(np.clip((self.s_path[after] - j - 2.0) / over, 0.0, 1.0)))
        tail, tail_i = tail + right * shift[:, None], tail_i + right * shift[:, None]
      joined, _ = resample(np.vstack([keep, tail]))
      joined_i, _ = resample(np.vstack([old_intent, tail_i]))
      n = min(len(joined), len(joined_i))
      th = headings(joined_i[:n])
      off = ((joined[:n] - joined_i[:n]) * np.stack([np.sin(th), -np.cos(th)], axis=1)).sum(axis=1)
      self._derive(route, joined_i[:n], off, t)
      self.changes = self._changes(self.keys)
      self.splices.append({"t": round(t, 3), "cut": round(cut, 1), "gap": round(float(d), 2), "pos": [round(float(keep[-1][0]), 1), round(float(keep[-1][1]), 1)]})
      self._event("mapdrive", t, phase=self.phase, splice=self.splices[-1])

  # *** lane changes the player asks for ***

  def ask_lane(self, side: str, t: float) -> str:
    """The player's arrow key asking for a change into the lane beside (`side`: 'left' or 'right'): 'ok' (made once
    that lane is clear, within ASK_WAIT s: what becomes of it comes in take_prompts(): 'lane' as it starts, 'wait' while
    the lane beside isn't clear, 'blocked' as it gives up, or why it can't), or why it can't now: 'busy' (a lane change
    under way or signalled), 'no_lane' (no lane our way that side, or the car's lane isn't known), 'lane_no' (nav would
    want back out of it soon: it turns off, ends, or the route needs this lane). The other side's arrow while one waits
    cancels it ('cancel')."""
    if self.asked is not None:
      if self.asked["side"] == side:
        return "ok"
      self.asked = None
      return "cancel"
    why = self._ask_check(side)
    if why is None:
      self.asked = {"side": side, "t": t, "waited": False}
    return why or "ok"

  def _ask_check(self, side: str) -> str | None:
    route = self.route
    if route is None or self.path is None or self.phase not in ASK_PHASES or not route.on_road():
      return "no_lane"
    v = self.last_v
    if any(s0 - self.style["signal_lead"] * max(v, 3.0) <= self.s <= s1 for s0, s1, *_ in self.changes):
      return "busy"
    lane = route.lane()
    if lane is None or lane[0] < 0:
      return "no_lane"
    i = lane[0] + (-1 if side == "left" else 1)
    if not 0 <= i < lane[1]:
      return "no_lane"
    reach = route.at + max(ASK_KEEP, ASK_KEEP_S * v)
    keys = self.nav_keys(route, [i, lane[1]])
    for (s0, l0), (s1, l1) in zip(keys, keys[1:], strict=False):
      if s0 > reach:
        break
      if s1 - s0 > EPS and (l1 - l0) * (lane[0] - i) > 0:
        return "lane_no"  # a change back towards the car's lane (not a renumbering, which is a step)
    return None

  def _lane_clear(self, side: str, state: dict) -> bool:
    """Whether the lane beside (`side`) is clear for a lane change (_lane_busy)."""
    return self._lane_busy(side, state) is None

  def _lane_busy(self, side: str, state: dict) -> str | None:
    """What keeps the lane beside (`side`) from being clear for a lane change, "beside" or "closing", else None: the
    blind-spot monitor's zone and closing rule with a lane change's margins (LaneGapTune, gta5_blindspot.lane_occupied)."""
    g = self.gap_tune
    v = float(state.get("vEgo") or 0.0)
    g.values["behind"] = min(max(g.behind_s * v, g.behind_min), LaneGapTune.DEFAULTS["behind"])
    return lane_occupied(state, side, g)

  # *** planned lane changes wait for their lane ***

  def _lc_gate(self, route, state: dict, t: float, v: float):
    """The next lane change on the path (planned, or asked for), from LC_GATE_T s before it starts until LC_COMMIT of
    the way across, made only while its lane is clear (_lane_clear). Otherwise it's put off (_deferred_keys), started
    again LC_DEFER_S s on from the car and planned again from where the car is: one under way goes back to its lane;
    meanwhile, with one alongside, the car drops back to LC_DROP_BACK of its speed to let it by.
    Where it can't be put off (it would run into the plan's next key or a stop line), the car stops short of it
    (_longitudinal) until its lane clears, the trip ending after LC_LATE_WAIT s. A lane_gap event as it starts to wait
    ("wait"), goes back ("back"), can wait no more ("late") and goes ("go")."""
    lw = self.lc_wait
    for s0, s1, side, shift in self.changes:
      if s1 <= self.s:
        continue
      if self.s < s0 - max(v, 3.0) * LC_GATE_T or self.s > s0 + LC_COMMIT * (s1 - s0):
        if lw is not None and self.s > s0:  # under way past the point of no return: it went
          self._event("lane_gap", t, phase="go", side=side, waited=round(t - lw["since"], 1), defers=lw["defers"])
          self.lc_wait = None
        return
      why = self._lane_busy(side, state) or self._lane_window(s0, s1, side, shift, v, t)
      if why is None and lw is not None and self.s <= s0:
        lw["clear"] = lw.get("clear") or t
        if t - lw["clear"] < LC_CLEAR_S:
          why = lw["why"]  # one reading clear after a busy one isn't a gap yet
      elif lw is not None:
        lw["clear"] = None
      if why is None:
        if lw is not None:
          self._event("lane_gap", t, phase="go", side=side, waited=round(t - lw["since"], 1), defers=lw["defers"])
          self.lc_wait = None
        return
      started = self.s > s0
      if lw is None:
        lw = self.lc_wait = {"since": t, "side": side, "defers": 0, "late": False, "s0": s0, "v0": v}
        self._event("lane_gap", t, phase="wait", side=side, why=why, started=started)
      lw["why"] = why
      keys = self._deferred_keys(route, s0, route.at + max(v * LC_DEFER_S, LC_DEFER_MIN), v)
      if keys is not None and self._replan_keys(route, state, t, keys):
        lw.update(defers=lw["defers"] + 1, late=False)
        if started:
          self._event("lane_gap", t, phase="back", side=side, why=why)
        return
      if self.phase == "abort":
        return
      if not started and not lw.get("branch"):
        other = self._other_branch(route, s0, v)
        if other is not None and self._replan_keys(route, state, t, other[0]):
          # into a fork's branch, no gap in time: on in our lane, along the other branch, the route asked for again
          self.avoid = [other[1]]
          lw["branch"] = True
          self._event("lane_gap", t, phase="other_branch", side=side, why=why, fork=round(other[2] - route.at, 1))
          return
      if not lw["late"] and not started:
        lw.update(late=True, s0=s0)
        self._event("lane_gap", t, phase="late", side=side, why=why)
      if t - lw["since"] > LC_LATE_WAIT:
        self._abort(f"no gap in the lane to the {side} in {LC_LATE_WAIT:.0f} s", t, state.get("pos"))
      return

  def _lane_window(self, s0: float, s1: float, side: str, shift: float, v: float, t: float) -> str | None:
    """What keeps a lane change from path m s0 to s1 (`shift` m across) from its lanes, beyond what the blind-spot
    rule sees (_lane_busy: alongside, or closing from behind, in the lane beside): from the vehicles' predicted motion
    (_predict), one in any lane it moves into, at any time from now until LC_AFTER_S s after it ends (the car at its
    speed), nearer our body along the lane than the gap a change leaves (LC_GAP_M plus LC_GAP_T s of the follower's
    speed, and LC_GAP_CLOSE s of how much faster one behind is, or the room to slow to the speed of one ahead at
    LC_CUT_DECEL): "ahead" (we'd cut in on one), "behind" (one would come up on us) or "beside", "into" where a far lane
    of a change across more than one is busy; else None."""
    if not self.cars or s1 - s0 < 1.0:
      return None
    n = max(int(round(abs(shift) / LANE_W)), 1)
    sign = 1.0 if side == "right" else -1.0
    vv = max(v, 3.0)
    times = np.arange(0.0, max(s1 - self.s, 0.0) / vv + LC_AFTER_S + 1e-6, 0.25)
    ours = self.s + vv * times
    moved_us = n * LANE_W * quintic((ours - s0) / (s1 - s0))  # our path's move across, at our place at each time
    for car in self.cars:
      if car.get("ped"):
        continue
      corners = self._predict(car, t, times, speed_up=True)
      along, right = self._place(corners.reshape(-1, 2), LEAD_AHEAD + 40.0, behind=60.0, off_max=ALONG_OFF + n * LANE_W)
      along, right = along.reshape(len(times), 4), right.reshape(len(times), 4)
      vi = float(np.hypot(*car["vel"]))
      for k in range(len(times)):
        if np.isnan(along[k]).any():
          continue
        # its body's place across, in lanes from where our change starts (our path itself moves across at its place)
        moved = n * LANE_W * quintic((along[k] - s0) / (s1 - s0))
        lat = sign * (right[k] + sign * moved)
        lo, hi = float(lat.min()), float(lat.max())
        lane = next((m for m in range(1, n + 1) if lo < (m + 0.5) * LANE_W - 0.3 and hi > (m - 0.5) * LANE_W + 0.3), None)
        if lane is None or moved_us[k] + self.half_width < (lane - 0.5) * LANE_W:
          continue  # in none of its lanes, or we're not into that lane yet then
        front, rear = ours[k] + self.front, ours[k] - self.rear
        a0, a1 = float(along[k].min()), float(along[k].max())
        gap_m = LC_GAP_SLOW if max(v, vi) < LC_SLOW else LC_GAP_M  # creeping into a queue, a metre is room
        if a0 >= front:  # behind it once in, slowing to its speed (if we must) no harder than LC_CUT_DECEL
          gap = a0 - front
          need = gap_m + LC_GAP_T * vv + max(vv - vi, 0.0) ** 2 / (2 * LC_CUT_DECEL)
          why = "ahead"
        elif a1 <= rear:
          need, gap, why = gap_m + LC_GAP_T * vi + LC_GAP_CLOSE * max(vi - vv, 0.0), rear - a1, "behind"
        else:
          need, gap, why = 1.0, -1.0, "beside"
        if gap < need:
          return "into" if lane > 1 else why
    return None

  def _deferred_keys(self, route, s0: float, start: float, v: float | None = None) -> list[tuple[float, float]] | None:
    """The plan's keys with the lane change starting at path m s0 put off to start at route m `start`, its length kept
    but shortened, where what comes after leaves less room, to no less than _lc_least (a lane change's least time at
    the car's speed `v`); the keys after it in the lane it moves into (section boundaries) go with it. A change after
    it in the same run of lanes (nothing but holds between) it would come within LC_SETTLE_S of is put off with it, as
    squeezing both into one sweep across two lanes is what a lane-by-lane gap check can't keep clear. The run ends at
    the plan's next renumbering or the route's end, and not across a stop line or a city junction node it was before.
    None where it can't be put off."""
    keys = self.keys
    v = self.last_v if v is None else v
    rs0 = float(np.interp(s0, self.s_path, self.route_s))
    found = chains(keys)
    if not found:
      return None
    first = min(range(len(found)), key=lambda n: abs(keys[found[n][0]][0] - rs0))
    if start <= keys[found[first][0]][0]:
      return None
    # the run: this change and those after it with only holds in their lanes between
    run = [found[first]]
    for a, b in found[first + 1:]:
      pb = run[-1][1]
      if any(abs(keys[m][1] - keys[pb][1]) > EPS or keys[m + 1][0] - keys[m][0] <= EPS for m in range(pb, a)):
        break
      run.append((a, b))
    lane1 = keys[run[-1][1]][1]
    j = run[-1][1] + 1  # past the keys that only hold the last one's lane on: the next is another change's start, or a renumbering
    while j < len(keys) and abs(keys[j][1] - lane1) < EPS and keys[j][0] - keys[j - 1][0] > EPS:
      j += 1
    if j >= len(keys):
      limit = route.length
    elif abs(keys[j][1] - lane1) >= EPS and keys[j][0] - keys[j - 1][0] > EPS:
      limit = keys[j - 1][0]  # a ramp to another lane begins at the key before it
    else:
      limit = keys[j][0]  # a renumbering step
    # nor across a stop line or a city junction it was before (a freeway's nodes are its slip roads' merges and exits)
    k1 = keys[run[0][1]][0]
    fast = route.classes[segment_of(route, k1)] in ("motorway", "trunk") if getattr(route, "classes", None) else False
    forks = [f.along for f in getattr(route, "forks", []) if f.keep]  # nor past a fork (its branch is the change's reason)
    after = [x for x in list(route.stops) + ([] if fast else list(route.junctions)) + forks if x > k1 - EPS]
    limit = min([limit] + [x - CROSSING_CLEAR for x in after]) - EPS
    settle = max(v, 5.0) * LC_SETTLE_S
    lengths = [keys[b][0] - keys[a][0] for a, b in run]
    least = [min(self._lc_least(abs(keys[b][1] - keys[a][1]), v), length) for (a, b), length in zip(run, lengths, strict=True)]
    placed = []  # (chain, its new start, its new end)
    s = start
    for n, ((a, b), length) in enumerate(zip(run, lengths, strict=True)):
      if n and s <= keys[a][0] + EPS:
        break  # not reached: it stays where it was, and so do those after it
      room = limit - s - sum(least[n + 1:len(run)]) - settle * (len(run) - n - 1)
      end = s + min(length, room)
      if end - s < least[n] - EPS:
        return None
      placed.append(((a, b), s, end))
      s = end + settle
    rest = run[len(placed)][0] if len(placed) < len(run) else j
    out = list(keys[:run[0][0]])
    for n, ((a, b), s_a, end) in enumerate(placed):
      k0, scale = keys[a][0], (end - s_a) / max(keys[b][0] - keys[a][0], EPS)
      out += [(s_a + (k[0] - k0) * scale, k[1]) for k in keys[a:b + 1]]
      upto, nxt = (placed[n + 1][0][0], placed[n + 1][1]) if n + 1 < len(placed) else (rest, math.inf)
      out += [k for k in keys[b + 1:upto] if end + EPS < k[0] < nxt - EPS]
    return out + list(keys[rest:])

  def _other_branch(self, route, s0: float, v: float) -> tuple[list, tuple, float] | None:
    """For the lane change starting at path m s0 that can't be put off any more, where it and the changes of its run
    lead into a fork's branch on a fast road (motorway or trunk) at least BRANCH_ROOM_S s ahead: the keys that stay in
    our lane (to the route's end: the new route says what lies past the fork), the link BRANCH_PAST m past the fork on
    the branch given up, for the route to keep off, and the fork's place (m along); None otherwise."""
    keys = self.keys
    rs0 = float(np.interp(s0, self.s_path, self.route_s))
    found = chains(keys)
    if not found or not getattr(route, "classes", None):
      return None
    first = min(range(len(found)), key=lambda n: abs(keys[found[n][0]][0] - rs0))
    a = found[first][0]
    if route.classes[segment_of(route, keys[a][0])] not in ("motorway", "trunk", "motorway_link", "trunk_link"):
      return None
    fork = next((f for f in route.forks if f.keep and keys[a][0] < f.along), None)
    if fork is None or fork.along - route.at < BRANCH_ROOM_S * max(v, 5.0):
      return None
    lane0 = keys[a][1]
    # in our lane on: what lies past the fork is the new route's to say, and it comes before the car gets there
    new = list(keys[:a]) + [(keys[a][0], lane0), (max(route.length, keys[-1][0]), lane0)]
    k = segment_of(route, min(fork.along + BRANCH_PAST, route.length - 1.0))
    link = (tuple(float(c) for c in route.points[k, :2]), tuple(float(c) for c in route.points[k + 1, :2]))
    return new, link, float(fork.along)

  def _lc_least(self, lanes: float, v: float) -> float:
    """m a lane change across `lanes` lanes put off takes at the least: MIN_LC_M a lane, and LC_LEAST_SHARE of its
    time (lc_seconds) at the car's speed."""
    n = max(lanes, 1.0)
    return max(MIN_LC_M * n, LC_LEAST_SHARE * lc_seconds(self.style, n) * max(v, 5.0))

  def _replan_keys(self, route, state: dict, t: float, keys: list[tuple[float, float]]) -> bool:
    """The path for `keys` from where the car is (as the first plan's: from its lane line, joined as a lane change
    would), the plan's keys and changes with it; False (an abort, from _build) where there's no line."""
    lane = route.lane() if route.on_road() else None
    if lane is None or lane[0] < 0:
      lane = [int(round(lane_at(keys, route.at))), max(int(round(lane_at(keys, route.at))) + 1, 1)]
    if self._build(route, keys, lane, state, t) is None:
      return False
    self.keys, self.changes = keys, self._changes(keys)
    self.plan = {**self.plan, "keys": [(round(s, 2), round(lane_, 3)) for s, lane_ in keys]}
    return True

  def _ask_step(self, route, state: dict, t: float):
    """The lane change asked for: made once the lane beside is clear, given up after ASK_WAIT s."""
    a = self.asked
    side = a["side"]
    if t - a["t"] > ASK_WAIT:
      self.asked = None
      self.prompts.append(("blocked", side))
      return
    if not self._lane_clear(side, state):
      if not a["waited"]:
        a["waited"] = True
        self.prompts.append(("wait", side))
      return
    self.asked = None
    why = self._ask_check(side)
    if why is not None:
      self.prompts.append((why, side))
      return
    self._splice(route, state, t, shift=-1 if side == "left" else 1)
    if self.phase != "abort":
      self.prompts.append(("lane", side))

  def _same_roads(self, route) -> float:
    """Path m (the old path's) to where a new route stops running along the old route's line on from the car (within
    SAME_LINE m of it), 0 where it doesn't from the car."""
    old, new = self.route, route
    k0 = max(int(np.searchsorted(new.along, new.at, side="right")) - 1, 0)
    pts = new.points[k0:, :2]
    if old is None or len(pts) < 2 or len(old.points) < 2:
      return 0.0
    a, ab = old.points[:-1, :2], np.diff(old.points[:, :2], axis=0)
    ab2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9)
    along = None
    for p in pts:
      rel = p - a
      tt = np.clip(np.einsum("ij,ij->i", rel, ab) / ab2, 0.0, 1.0)
      d = np.hypot(*(rel - ab * tt[:, None]).T)
      i = int(np.argmin(d))
      if d[i] > SAME_LINE or (along is not None and old.along[i] + tt[i] * math.sqrt(ab2[i]) < along - SAME_LINE):
        break
      along = float(old.along[i] + tt[i] * math.sqrt(ab2[i]))
    if along is None or along <= old.at:
      return 0.0
    return float(np.interp(along, self.route_s, self.s_path))

  # *** vehicles ***

  def _vehicles(self, state: dict, t: float):
    """The vehicles and pedestrians around from the plugin's nearby, into the world by the pose of the state that
    brought them (it reports every 0.1 s, the state comes each frame, and its places are from the car where it was then);
    pedestrians are kept with the vehicles, flagged "ped"."""
    nb = state.get("nearby")
    if nb is None:
      self.cars = None
      return
    self.reach = (float(nb.get("ahead") or NEARBY_AHEAD), float(nb.get("side") or NEARBY_SIDE))
    self.peds_reported = "p" in nb
    rows = (nb.get("v") or [], nb.get("p") or [])
    if self.cars is not None and rows == self.cars_rows:
      self.traffic.again(self.cars, t)
      return
    self.cars_rows = rows
    h = math.radians(float(state.get("heading") or 0.0))
    right, fwd = np.array([math.cos(h), math.sin(h)]), np.array([-math.sin(h), math.cos(h)])
    pos = np.asarray(state["pos"][:2], float)
    cars = []
    for v in rows[0]:
      if len(v) < 9:
        continue
      x, y, rel, vx, vy, mnx, mxx, mny, mxy = (float(a) for a in v[:9])
      c, s = math.cos(math.radians(rel)), math.sin(math.radians(rel))
      local = np.array([(x + cx * c - cy * s, y + cx * s + cy * c) for cx, cy in ((mnx, mny), (mxx, mny), (mxx, mxy), (mnx, mxy))])
      cars.append({"c": pos + np.outer(local[:, 0], right) + np.outer(local[:, 1], fwd), "vel": right * vx + fwd * vy, "t": t,
                   "driven": len(v) < 10 or bool(v[9]), **({"handle": int(v[10])} if len(v) > 10 else {}),
                   **({"at_light": bool(v[11])} if len(v) > 11 else {})})
    body = np.array([(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]) * PED_HALF
    for x, y, vx, vy in (r[:4] for r in rows[1] if len(r) >= 4):
      cars.append({"c": pos + right * float(x) + fwd * float(y) + body, "vel": right * float(vx) + fwd * float(vy), "t": t,
                   "driven": None, "ped": True})
    self.traffic.update(cars, t)
    self.cars = cars

  def _place(self, pts: np.ndarray, ahead: float, path: np.ndarray | None = None, behind: float = 5.0,
             off_max: float = LEAD_OFF) -> tuple[np.ndarray, np.ndarray]:
    """Points' (m along the path, or `path` of the same length, m right of it) on the path from `behind` m back of the
    car to `ahead` m on; NaN beyond off_max off it or off either end."""
    sp = self.s_path
    path = self.path if path is None else path
    lo = max(int(np.searchsorted(sp, self.s - behind)) - 1, 0)
    hi = min(int(np.searchsorted(sp, self.s + ahead)) + 1, len(sp) - 1)
    a, ab = path[lo:hi], path[lo + 1:hi + 1] - path[lo:hi]
    if not len(a):
      return np.full(len(pts), np.nan), np.full(len(pts), np.nan)
    ab2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9)
    rel = pts[:, None, :] - a[None, :, :]
    tt = np.einsum("nij,ij->ni", rel, ab) / ab2
    tc = np.clip(tt, 0.0, 1.0)
    d = np.hypot(*(rel - ab[None] * tc[..., None]).transpose(2, 0, 1))
    i = np.argmin(d, axis=1)
    n = np.arange(len(pts))
    seg = np.sqrt(ab2[i])
    along = sp[lo + i] + tc[n, i] * seg
    right = (rel[n, i, 0] * ab[i, 1] - rel[n, i, 1] * ab[i, 0]) / seg
    off = (d[n, i] > off_max) | ((i == 0) & (tt[n, i] < 0.0)) | ((i == len(a) - 1) & (tt[n, i] > 1.0))
    along[off], right[off] = np.nan, np.nan
    return along, right

  def _car_now(self, car: dict, t: float, ahead: float = 0.0) -> np.ndarray:
    """A vehicle's corners `ahead` s after now, straight on at its velocity from its reading."""
    return car["c"] + car["vel"] * (t - car["t"] + ahead)

  def _predict(self, car: dict, t: float, times: np.ndarray, speed_up: bool = False) -> np.ndarray:
    """A vehicle's corners at each of `times` s after now (gta5_traffic's prediction): one running along the road our
    route runs on (its heading within ALONG_ANGLE of the route's there, or of its reverse) and turning as the road does
    (its yaw rate within TURN_OFF of the path's curvature at its speed, ALONG_BEND m about its place), at no more than
    ALONG_OFF beside our path, goes on along the route's line at its offset from it and its speed (its braking carried
    on to a stop); any other on at its yaw rate and speed (gta5_traffic.ctra); a pedestrian straight on. With speed_up
    (a lane change's look behind), one along the road speeding up goes on doing so (at its tracked rate, SPEED_UP_MAX at
    most) to SPEED_UP_TO. Along the route's line, not our path, which moves across lanes in a lane change. (One going
    straight on where our route turns isn't along the road it leaves for: off the road ahead it isn't along the route at
    all, and on it, once the route turns, its yaw rate no longer matches.)"""
    times = np.asarray(times, float)
    speed = float(np.hypot(*car["vel"]))
    if car.get("ped") or car.get("obst") or speed < 0.5 or self.path is None:
      return np.stack([self._car_now(car, t, dt) for dt in times])
    c0 = self._car_now(car, t)
    along, _ = self._place(c0.mean(axis=0)[None], LEAD_AHEAD + 40.0, behind=60.0, off_max=ALONG_OFF)
    if not math.isnan(along[0]):
      route = self.route
      u0 = float(np.interp(along[0], self.s_path, self.route_s))
      k = segment_of(route, u0)
      seg = route.points[k + 1, :2] - route.points[k, :2]
      th = math.atan2(seg[1], seg[0])
      hv = math.atan2(car["vel"][1], car["vel"][0])
      diff = abs(wrap_rad(hv - th))
      way = 1.0 if diff < math.radians(ALONG_ANGLE) else -1.0 if diff > math.pi - math.radians(ALONG_ANGLE) else 0.0
      # the road's bends about its place (it may be where a bend starts or ends)
      near = np.abs(self.s_path - along[0]) < ALONG_BEND
      bends = (self.kappa_path[near] if near.any() else np.zeros(1)) * speed * way
      if way and float(np.min(np.abs(float(car.get("yaw") or 0.0) - bends))) < TURN_OFF:
        mid = c0.mean(axis=0)
        w = float((mid[0] - route.points[k, 0]) * seg[1] - (mid[1] - route.points[k, 1]) * seg[0]) / max(float(np.hypot(*seg)), 1e-6)
        acc = min(float(car.get("accel") or 0.0), 0.0)
        go = np.minimum(speed * times + 0.5 * acc * times ** 2, speed * speed / (2 * -acc)) if acc < -0.1 else speed * times
        up = min(float(car.get("accel") or 0.0), SPEED_UP_MAX)
        if speed_up and up > 0.1 and speed < SPEED_UP_TO:  # one speeding up goes on doing so, to SPEED_UP_TO
          t_up = (SPEED_UP_TO - speed) / up
          go = np.where(times < t_up, speed * times + 0.5 * up * times ** 2, speed * t_up + 0.5 * up * t_up ** 2 + SPEED_UP_TO * (times - t_up))
        reach = (route.length - u0) if way > 0 else u0  # along the route's line to its end
        length, width = size(c0)
        out = []
        for d in go:
          u = u0 + way * min(d, reach)
          kk = segment_of(route, u)
          sg = route.points[kk + 1, :2] - route.points[kk, :2]
          h = math.atan2(sg[1], sg[0])
          p = route_point(route, u) + np.array([math.sin(h), -math.cos(h)]) * w
          h += 0.0 if way > 0 else math.pi
          if d > reach:  # straight on past the route's end
            p = p + (d - reach) * np.array([math.cos(h), math.sin(h)])
          out.append(rect(p, h, length, width))
        return np.stack(out)
    return ctra(car, t, times)

  @staticmethod
  def _outline(c: np.ndarray) -> np.ndarray:
    """A body's corners and the middles of its sides and its middle: the points placed along the path."""
    return np.vstack([c, (c + np.roll(c, -1, axis=0)) / 2, c.mean(axis=0)[None]])

  def _obstacles(self, state: dict, t: float):
    """The static things the plugin's probes hit ahead (nearby's "obst", read every 0.1-0.2 s: points from the car where
    it was then), each as a small body in the world, standing; none where the plugin doesn't probe."""
    rows = (state.get("nearby") or {}).get("obst")
    if rows == self.obst_rows:
      return
    self.obst_rows = rows
    h = math.radians(float(state.get("heading") or 0.0))
    right, fwd = np.array([math.cos(h), math.sin(h)]), np.array([-math.sin(h), math.cos(h)])
    pos = np.asarray(state["pos"][:2], float)
    body = np.array([(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]) * OBST_HALF
    self.obst = [{"c": pos + right * float(r[0]) + fwd * float(r[1]) + body, "vel": np.zeros(2), "t": t, "driven": None, "obst": True}
                 for r in rows or [] if len(r) >= 2]

  def _lead(self, t: float) -> tuple[float, float, dict, float] | None:
    """The nearest vehicle, pedestrian or static thing in our way along the path ahead (its body within LEAD_SIDE of
    ours, PED_SIDE for a pedestrian, OBST_SIDE for a static thing, or across the path):
    (m from our front bumper to it, its speed along the path, it, m right of the path its body comes nearest); None."""
    if not self.cars and not self.obst:
      return None
    best = None
    far = self.s + self.front + LEAD_AHEAD
    for car in (self.cars or []) + self.obst:
      pts = self._outline(self._car_now(car, t))
      along, right = self._place(pts, LEAD_AHEAD + self.front)
      ok = ~np.isnan(along)
      if not ok.any():
        continue
      side_m = PED_SIDE if car.get("ped") else OBST_SIDE if car.get("obst") else LEAD_SIDE
      near = ok & (np.abs(np.nan_to_num(right, nan=99.0)) < self.half_width + side_m)
      across = ok.sum() >= 2 and np.nanmin(right) < 0.0 < np.nanmax(right) and np.nanmax(along) - np.nanmin(along) < 12.0
      if not near.any() and not across:
        continue
      start = float(np.min(along[near])) if near.any() else float(np.nanmin(along))
      if start < self.s + self.front - LEAD_BESIDE or start > far:
        continue  # beside or behind us: no braking keeps clear of it
      gap = start - self.s - self.front
      j = int(np.clip(np.searchsorted(self.s_path, start), 0, len(self.s_path) - 1))
      th = self.theta[j]
      v_along = max(float(car["vel"] @ np.array([math.cos(th), math.sin(th)])), 0.0)
      side = 0.0 if across else float(right[near][np.argmin(np.abs(right[near]))])
      car["across_way"] = self._across(car, th)
      car["crosswise"] = car["across_way"] and car["driven"] and float(np.hypot(*car["vel"])) <= LEAD_MOVING
      if gap > 2.0 and self._clears(car, t, th, gap):
        continue  # crossing our way, gone before we get there: the crossing check's to judge, no lead to brake for
      if best is None or gap < best[0]:
        best = (gap, v_along, car, side)
    return best

  def _clears(self, car: dict, t: float, th: float, gap: float) -> bool:
    """Whether a vehicle moving across our way (heading CROSS_ANGLE or more off the path's `th`, faster than
    CROSS_MOVING) `gap` m ahead of our front is out of our way (its body, by _predict, no longer within LEAD_SIDE of
    ours along the path) CROSS_GAP_BEHIND s before we'd reach it at our speed."""
    speed = float(np.hypot(*car["vel"]))
    if car.get("ped") or car.get("obst") or speed < CROSS_MOVING:
      return False
    angle = abs(wrap(math.degrees(math.atan2(car["vel"][1], car["vel"][0]) - th)))
    if angle < CROSS_ANGLE or angle > 180.0 - CROSS_ANGLE:
      return False
    reach = gap / max(self.last_v, 1.0) - CROSS_GAP_BEHIND
    if reach <= 0.0:
      return False
    times = np.arange(0.0, reach + 1e-6, 0.25)
    corners = self._predict(car, t, times)
    _, right = self._place(self._outline(corners[-1]), LEAD_AHEAD + self.front)
    return bool(np.all(np.isnan(right) | (np.abs(np.nan_to_num(right, nan=99.0)) >= self.half_width + LEAD_SIDE)) and
                not (np.nanmin(right, initial=1.0) < 0.0 < np.nanmax(right, initial=-1.0)))

  @staticmethod
  def _across(car: dict, th: float) -> bool:
    """Whether a vehicle's body heads across our way (side on or head on to the path's heading `th`), not along it."""
    if car.get("ped"):
      return False
    fwd = car["c"][3] - car["c"][0]
    return abs(wrap(math.degrees(math.atan2(fwd[1], fwd[0]) - th))) > ACROSS_ANGLE

  def _crosswise(self, car: dict, th: float) -> bool:
    """Whether a vehicle is a driven one standing still across our way (as one waiting at a crossing's line, or swinging
    wide through a turn), not one going our way in our lane: an obstacle to pass or wait for while it moves, not a lead
    to queue behind."""
    return bool(car["driven"]) and float(np.hypot(*car["vel"])) <= LEAD_MOVING and self._across(car, th)

  def _visible(self) -> float:
    """m of path ahead of our front bumper inside the box the plugin reports vehicles within (reach ahead of our
    origin, to either side)."""
    ahead, side = self.reach
    sp = self.s_path
    i0 = int(np.searchsorted(sp, self.s))
    i1 = int(np.searchsorted(sp, self.s + ahead + 1.0))
    if i1 <= i0:
      return 0.0
    p0 = np.array([np.interp(self.s, sp, self.path[:, 0]), np.interp(self.s, sp, self.path[:, 1])])
    th = float(np.interp(self.s, sp, self.theta))
    f, r = np.array([math.cos(th), math.sin(th)]), np.array([math.sin(th), -math.cos(th)])
    q = self.path[i0:i1] - p0
    out = ((q @ f) > ahead) | (np.abs(q @ r) > side)
    k = int(np.argmax(out)) if out.any() else len(q)
    end = sp[i0 + k - 1] if k > 0 else self.s
    return max(float(end) - self.s - self.front - RANGE_BODY, 0.0) if out.any() else max(ahead - self.front - RANGE_BODY, 0.0)

  def _crossing(self, t: float, v: float) -> tuple[float, dict] | None:
    """A moving vehicle whose way (straight on at its speed, CROSS_T s) crosses ours near a junction or turn ahead
    within CROSS_GAP_AHEAD s after we'd be through or CROSS_GAP_BEHIND s before we'd be there (its way by
    _predict), or a walking pedestrian's (PED_T s, within PED_SIDE of our body) anywhere
    along the path: (m from our front bumper to where we'd stop for it, details); None. In a junction, slow (_in_junction):
    the nearest lane's crosser held for, not the farthest's: one whose way meets ours beside us held where we are, those
    up to CROSS_T_IN s off (clearing the junction from a standstill takes longer than CROSS_T), and one slowed or stopped
    beside our way (_slowed_across)."""
    if not self.cars:
      return None
    inside = v < CROSS_SLOW and self._in_junction()
    # s for the car to go on from here: speeding up from v at the style's accel, no faster than the profile
    sp = self.s_path
    i0 = int(np.searchsorted(sp, self.s))
    i1 = min(int(np.searchsorted(sp, self.s + LEAD_AHEAD + self.rear)) + 1, len(sp))
    ds = np.maximum(sp[i0:i1] - self.s, 0.0)
    speed = np.maximum(np.minimum(self.v_static[i0:i1], np.sqrt(v * v + 2 * max(self.style["accel"], 0.5) * ds)), 1.0)
    secs = np.concatenate(([0.0], np.cumsum(np.diff(ds) / (0.5 * (speed[1:] + speed[:-1])))))

    lag = CROSS_START_LAG if v < 1.0 else 0.0

    def when(d: float) -> float:
      return float(np.interp(d, ds, secs)) + lag if len(ds) > 1 else math.inf
    best = None
    for car in self.cars:
      ped = bool(car.get("ped"))
      speed = float(np.hypot(*car["vel"]))
      if speed < (PED_MOVING if ped else CROSS_MOVING):
        if inside and not ped and car["driven"]:
          slowed = self._slowed_across(car, t)
          if slowed is not None and (best is None or slowed[0] < best[0]):
            best = slowed
        continue
      times = np.arange(0.0, (PED_T if ped else CROSS_T_IN if inside else CROSS_T) + 1e-6, 0.25)
      along, right = self._place(np.concatenate([self._outline(c) for c in self._predict(car, t, times)]), LEAD_AHEAD)
      side = self.half_width + (PED_SIDE if ped else LEAD_SIDE)
      m = (~np.isnan(along) & (np.abs(np.nan_to_num(right, nan=99.0)) < side)).reshape(len(times), -1)
      along = along.reshape(len(times), -1)
      hits = [(float(times[k]), float(np.min(along[k][m[k]])), float(np.max(along[k][m[k]]))) for k in range(len(times)) if m[k].any()]
      if not hits:
        continue
      first = min(h[1] for h in hits)
      if first < self.s + self.front - LEAD_BESIDE:
        if not (inside and not ped and first > self.s - self.rear):
          continue  # in our way already: a lead
        # in a junction, its way meets ours beside us: held where we are, not crept on across its lane
      elif not (ped or self._near_crossing(first)):
        continue  # not where ways cross
      j = int(np.clip(np.searchsorted(self.s_path, first), 0, len(self.s_path) - 1))
      th = self.theta[j]
      angle = abs(wrap(math.degrees(math.atan2(car["vel"][1], car["vel"][0]) - th)))
      if angle < CROSS_ANGLE:
        continue  # going our way: a lead once it's in it
      t_in, t_out = min(h[0] for h in hits), max(h[0] for h in hits)
      last = max(h[2] for h in hits)
      us_in = when(first - self.s - self.front)
      us_out = when(last - self.s + self.rear)
      if us_in > t_out + CROSS_GAP_BEHIND or us_out < t_in - CROSS_GAP_AHEAD:
        continue  # it's gone before we're there, or we're through before it comes
      d = first - self.s - self.front - CROSS_STOP
      if not ped and d + CROSS_STOP - 0.5 < v * RANGE_LAG + v * v / (2 * DECEL_REAL):
        continue  # too late to stop short of it: on through (a pedestrian: as much braking as there's room for)
      if best is None or d < best[0]:
        best = (d, {"at": round(first, 1), "in": round(t_in, 1), "out": round(t_out, 1), "us": round(us_in, 1),
                    "angle": round(angle), "speed": round(speed, 1), **({"ped": True} if ped else {})})
    return best

  def _slowed_across(self, car: dict, t: float) -> tuple[float, dict] | None:
    """A driven vehicle slowed below CROSS_MOVING (as GTA's AI does for our nose in its lane) side on to our way within
    CROSS_BESIDE of our body ahead: (m from our front bumper to where we'd stop for it, details) as _crossing's; None."""
    pts = self._outline(self._car_now(car, t))
    along, right = self._place(pts, LEAD_AHEAD)
    near = ~np.isnan(along) & (np.abs(np.nan_to_num(right, nan=99.0)) < self.half_width + CROSS_BESIDE)
    if not near.any():
      return None
    first = float(np.min(along[near]))
    if first < self.s - self.rear:
      return None
    j = int(np.clip(np.searchsorted(self.s_path, first), 0, len(self.s_path) - 1))
    k = int(np.clip(np.searchsorted(self.s_path, first + SLOWED_TURN), j + 1, len(self.s_path)))
    fwd = car["c"][3] - car["c"][0]
    angles = np.abs((math.degrees(math.atan2(fwd[1], fwd[0])) - np.degrees(self.theta[j:k]) + 180.0) % 360.0 - 180.0)
    angle = float(angles[0])
    if float(angles.min()) < CROSS_ANGLE or float(angles.max()) > 180.0 - CROSS_ANGLE:
      return None  # along our way, or the way we're turning into (one queued in the road we turn into): a lead; or oncoming
    return (first - self.s - self.front - CROSS_STOP,
            {"at": round(first, 1), "angle": round(angle), "speed": round(float(np.hypot(*car["vel"])), 1), "slowed": True})

  def _red_ahead(self, t: float) -> bool:
    """Whether a signal not yet passed within LIGHT_LOOK m ahead was last read red (_light) within LIGHT_KEEP s."""
    return any(x["kind"] == "lights" and not x["done"] and x["s"] - self.s < LIGHT_LOOK and t - x.get("red_t", -math.inf) < LIGHT_KEEP
               for x in self.stops)

  def _node_of(self, x: dict) -> np.ndarray:
    """The world place of a stop line's junction: its first node within STOP_LINE_FAR past it, else 10 m past it."""
    after = [a for a in self.route.junctions if 0.0 <= a - x["along"] <= STOP_LINE_FAR]
    return route_point(self.route, after[0] if after else x["along"] + 10.0)

  def _light(self, x: dict, state: dict, t: float) -> str | None:
    """What the traffic about shows of the signal at stop line x, from LIGHT_LOOK m short of it: "red" where one of
    ours (going our way) ahead of our rear waits at a red light (its own flag, where the plugin sends one) or has stood
    at or just behind the line LIGHT_QUEUE s, or one crosses our way in its junction faster than LIGHT_CROSS, going
    straight (on its green; not turning); "green" where one of ours drives over the line faster than LIGHT_THROUGH, or one on the crossing road waits
    at its red by the junction, and none of that, but for one going through in our own lane (or any lane of ours where
    the route goes straight on), which outweighs a queue in the lane beside (a turn lane's own phase); each kept
    LIGHT_KEEP s. Never from those behind us, who wait because we do (the plugin's traffic.red counts them too, so isn't
    used). None where nothing shows: no traffic about, and the trip goes as without it. x keeps why, and whether any
    traffic was about."""
    gap = x["s"] - self.s - self.front
    if gap > LIGHT_LOOK or not self.cars:
      return None
    x["traffic"] = True
    red = green = None
    own_green = False
    j = int(np.clip(np.searchsorted(self.s_path, x["s"]), 0, len(self.s_path) - 1))
    th = float(self.theta[j])
    node = self._node_of(x)
    # going straight on through it, the lanes beside going through show our phase too (turning, only our own lane's)
    straight = not any(0.0 <= tr.dist - x["along"] < 40.0 and abs(tr.angle) > 30.0 for tr in (getattr(self.route, "_turns", None) or []))
    for car in self.cars:
      if car.get("ped") or not car["driven"]:
        continue
      mid = car["c"].mean(axis=0)
      speed = float(np.hypot(*car["vel"]))
      h = math.atan2(car["vel"][1], car["vel"][0]) if speed > 1.0 else body_heading(car["c"])
      ang = abs(wrap(math.degrees(h - th)))
      if ang < ALONG_ANGLE:
        along, right = self._place(mid[None], gap + self.front + 40.0, behind=40.0, off_max=2.0 * LANE_W)
        if math.isnan(along[0]) or along[0] < self.s - self.rear + 1.0:
          continue  # behind us: queued because we stand
        rel = float(along[0]) - x["s"]
        if (car.get("at_light") and rel < 3.0) or (-25.0 < rel < 3.0 and float(car.get("still") or 0.0) >= LIGHT_QUEUE):
          red = red or "waiting"
        elif -3.0 < rel < 20.0 and speed > LIGHT_THROUGH:
          green = "through"
          own_green = own_green or straight or abs(float(right[0])) < LANE_W / 2 + 1.0
      elif 50.0 < ang < 130.0 and float(np.hypot(*(mid - node))) < LIGHT_NODE:
        if speed > LIGHT_CROSS and abs(float(car.get("yaw") or 0.0)) < LIGHT_STRAIGHT:
          red = red or "cross"  # (one turning through the junction, from the road opposite or on its own red, shows nothing)
        elif car.get("at_light"):
          green = green or "cross_waiting"
    if red and not own_green:
      x["red_t"], x["why"] = t, red
    elif green:
      x["green_t"], x["why"], x["green_why"] = t, green, green
    # held at it, red stands LIGHT_HOLD_KEEP s (a gap in the cross traffic, or one waiting at its own red, isn't our
    # green), unless one of ours goes through after it
    keep = LIGHT_HOLD_KEEP if x.get("held") is not None else LIGHT_KEEP
    ours = x.get("green_why") == "through" and x.get("green_t", -math.inf) > x.get("red_t", -math.inf)
    if t - x.get("red_t", -math.inf) < keep and not ours:
      return "red"
    if t - x.get("green_t", -math.inf) < LIGHT_KEEP:
      return "green"
    return None

  def _before_us(self, x: dict, t: float) -> dict | None:
    """At a stop sign's line, stopped since x["t_stop"]: a driven vehicle waiting at its own line into the junction
    since before we stopped (standing within ORDER_NEAR of the junction's node, heading for it, not in our way along our
    path, which the lead and crossing checks see to, and having moved since it was seen: not one parked by the kerb
    with its driver idle): the first come goes first. None where there's none."""
    node = self._node_of(x)
    for car in self.cars or []:
      if car.get("ped") or not car["driven"] or car.get("obst"):
        continue
      still = float(car.get("still") or 0.0)
      if still <= 0.0 or t - still > x["t_stop"] - 0.3 or not car.get("moved", True):
        continue  # moving, came after us, or never moved since seen
      mid = car["c"].mean(axis=0)
      to = node - mid
      dist = float(np.hypot(*to))
      if not 3.0 < dist < ORDER_NEAR:
        continue
      h = body_heading(car["c"])
      if float(to @ np.array([math.cos(h), math.sin(h)])) < 0.5 * dist:
        continue  # not heading for the junction
      along, right = self._place(mid[None], ORDER_NEAR + 20.0, behind=10.0)
      if not math.isnan(along[0]) and abs(float(right[0])) < self.half_width + LEAD_SIDE + 1.0:
        continue  # in our way along our path: a lead's, or the crossing check's
      return car
    return None

  def _in_junction(self) -> bool:
    """Whether our front bumper is in a junction: past a stop line within JUNCTION_IN, or within JUNCTION_NODE of a node."""
    f = self.s + self.front
    if any(x["done"] and 0.0 <= f - x["s"] <= JUNCTION_IN for x in self.stops):
      return True
    along = float(np.interp(f, self.s_path, self.route_s))
    return any(abs(a - along) < JUNCTION_NODE for a in self.route.junctions)

  def _line_short_of(self, at: float, v: float) -> float | None:
    """m from our front bumper's place to where we'd stop for the signal or give way line not yet passed short of path m
    `at`, at its junction (within JUNCTION_IN), if there's room to stop there; None. (A stop sign's line has its own full
    stop.) Kept to the line, not the place, so a signal's state can drive it too."""
    for x in self.stops:
      if x["done"] or x["kind"] not in ("lights", "give_way") or not 0.0 <= at - x["s"] <= JUNCTION_IN:
        continue
      d = x["target"] - self.s
      if d + 1.0 >= v * STOP_LAG + v * v / (2 * DECEL_REAL):
        return d
    return None

  def _ped_kerb(self, t: float, v: float, look: float) -> tuple[float, dict] | None:
    """The speed (m/s) from which the car could stop CROSS_STOP short of where a pedestrian beside the path ahead would
    step into it (PED_KERB_DECEL after RANGE_LAG, but PED_KERB_MIN at the least), for those within PED_KERB of our
    body near a junction or turn (waiting to cross, a light we can't see may be theirs), or anywhere heading for our path;
    not those walking away from it, or we'd be past before they could reach it at PED_STEP. With the nearest such one's
    details; None."""
    best = None
    for car in self.cars or []:
      if not car.get("ped"):
        continue
      c = self._car_now(car, t).mean(axis=0)
      along, right = self._place(c[None], LEAD_AHEAD + self.front)
      a, r = float(along[0]), float(right[0])
      if math.isnan(a):
        continue
      d = a - self.s - self.front
      side = abs(r) - self.half_width - PED_HALF  # m from our body's side to theirs (nearer ones are leads too)
      if d < 0.0 or side + PED_HALF >= PED_KERB:
        continue
      j = int(np.clip(np.searchsorted(self.s_path, a), 0, len(self.s_path) - 1))
      th = self.theta[j]
      towards = -math.copysign(1.0, r) * float(car["vel"] @ np.array([math.sin(th), -math.cos(th)]))
      if towards < -PED_MOVING or not (self._near_crossing(a) or towards > PED_MOVING):
        continue  # walking away, or neither near a crossing nor heading for the path
      if v * max(side, 0.0) / PED_STEP > d + self.front + self.rear:
        continue  # past before they could step in
      k = PED_KERB_DECEL * RANGE_LAG
      # vp is the speed `look` m on
      cap = max(-k + math.sqrt(k * k + 2 * PED_KERB_DECEL * max(d - look - CROSS_STOP, 0.0)), PED_KERB_MIN)
      if best is None or cap < best[0]:
        best = (cap, {"gap": round(d, 1), "right": round(r, 2), "speed": round(float(np.hypot(*car["vel"])), 1),
                      "cap": round(cap, 1)})
    return best

  def _near_crossing(self, s: float) -> bool:
    """Whether a place on the path (m) is near a junction node or turn of the route."""
    along = float(np.interp(s, self.s_path, self.route_s))
    spots = list(self.route.junctions) + [tr.dist for tr in (getattr(self.route, "_turns", None) or [])]
    return any(abs(a - along) < CROSS_NEAR for a in spots)

  # *** passing a parked vehicle within our lane ***

  def _nudge_room(self) -> tuple[np.ndarray, np.ndarray]:
    """How far (m, right positive) each point of the path may move and keep our body within its lane, CLAMP_MARGIN in
    from its edges (_clamp's lane room, which the plan was kept to): (left bound, right bound), NaN where the lane is no
    guide (lane changes, jogs, tapers, about turns) or the plan has none."""
    n = len(self.path)
    if self.lane_room is None:
      return np.full(n, np.nan), np.full(n, np.nan)
    grid, centre, left, right, _ = self.lane_room
    g = np.round((self.route_s - grid[0]) / LANE_GRID).astype(int)
    inside = (g >= 0) & (g < len(grid))
    g = np.clip(g, 0, len(grid) - 1)
    off = np.where(inside, self.route_r - centre[g], np.nan)
    return -left[g] - off, right[g] - off

  def _fill_room(self, a: int, b: int):
    """The nudge's room (room_lo, room_hi) filled in over path points a to b, kept for the trip, where the plan's lane
    room has none (lane changes, jogs, tapers, a fractional lane, about junctions): from the road's cross-section where
    the path runs along its line, NUDGE_NODE from junction nodes and short of turns (where the map's sections describe
    the junction, not our lane), our body CLAMP_MARGIN within our lane and the road's edges, or within the lanes the path
    moves across (ours either way: a change's two, a jog's into the lane carrying on); across a stretch still without
    (NUDGE_BRIDGE, no turn in it), the lane's edges either side carried across. Never into a lane the path doesn't take,
    nor where a turn's path leaves the road's line."""
    a, b = max(a, 0), min(b, len(self.path0) - 1)
    todo = [i for i in range(a, b + 1) if not self.room_tried[i]]
    if not todo:
      return
    route, lanes = self.route, self.route.lanes
    nodes = np.asarray(route.junctions, float)
    turns = np.asarray(self._turn_alongs, float)
    forks = [(f.along, f.side) for f in route.forks if f.keep]
    room0 = self.half_width + CLAMP_MARGIN
    for i in todo:
      self.room_tried[i] = True
      rs, r = float(self.route_s[i]), float(self.route_r[i])
      near = self.route_r[max(i - NUDGE_ACROSS, 0):i + NUDGE_ACROSS + 1]
      across = float(near.max() - near.min()) > 1.0
      if not (across or np.isnan(self.room_lo[i])):
        continue
      if len(nodes) and np.abs(nodes - rs).min() < NUDGE_NODE:
        continue
      close = turns[np.abs(turns - rs) < CORNER_ZONE] if len(turns) else turns
      if len(close) and (close - rs).min() < NUDGE_NODE:
        continue  # past a turn's point, or about to reach it
      k = segment_of(route, rs)
      d = route.points[k + 1, :2] - route.points[k, :2]
      sec = lanes.section_at(rs, k)
      if sec is None or not sec.lanes:
        continue
      if abs(wrap(math.degrees(self.theta0[i] - math.atan2(d[1], d[0])))) > NUDGE_ALIGN * (2.0 if across else 1.0):
        continue
      # the lane the path is in, or those it moves across between
      r0, r1 = (float(near.min()), float(near.max())) if across else (r, r)
      hit = [sp for sp in sec.ours if sp.left < r1 + self.half_width and sp.right > r0 - self.half_width]
      if not across:
        hit = [sp for sp in hit if sp.left <= r <= sp.right] if len(hit) > 1 else hit
      if not hit or any(not NUDGE_SPAN[0] <= sp.right - sp.left <= NUDGE_SPAN[1] for sp in hit) or \
         any(abs(b_.left - a_.right) > 0.1 for a_, b_ in zip(hit, hit[1:], strict=False)):
        continue
      left, right = hit[0].left, hit[-1].right
      if not left <= r <= right:
        continue
      lo, hi = max(left, sec.edges[0]) + room0 - r, min(right, sec.edges[1]) - room0 - r
      own = next((sp for sp in hit if sp.left <= r <= sp.right), hit[0])
      gore = next((side for fa, side in forks if -FORK_BEFORE < rs - fa < FORK_AFTER), None)
      if gore == "right":  # the branch taken is the right one: no nearer the gore on our left than its _lane_room allows
        lo = max(lo, own.centre - FORK_ROOM - r)
      elif gore == "left":
        hi = min(hi, own.centre + FORK_ROOM - r)
      if lo < hi and np.isnan(self.room_lo[i]):
        self.room_lo[i], self.room_hi[i] = lo, hi
      elif lo < hi:  # the plan's own lane, widened to the lanes it moves across
        self.room_lo[i], self.room_hi[i] = min(lo, self.room_lo[i]), max(hi, self.room_hi[i])
    # a stretch still without, between two with, no turn in it
    sp, lo_, hi_ = self.s_path, self.room_lo, self.room_hi
    for g0, g1 in _runs(np.isnan(lo_[a:b + 1])):
      i0, i1 = a + g0, a + g1 - 1
      if i0 == 0 or i1 + 1 >= len(sp) or np.isnan(lo_[i0 - 1]) or np.isnan(lo_[i1 + 1]):
        continue
      if sp[i1 + 1] - sp[i0 - 1] > NUDGE_BRIDGE or abs(wrap(math.degrees(self.theta0[i1 + 1] - self.theta0[i0 - 1]))) > NUDGE_BRIDGE_DEG:
        continue
      if len(turns) and ((turns > self.route_s[i0 - 1]) & (turns < self.route_s[i1 + 1])).any():
        continue
      # the lanes' edges carried straight across (m right of the road's line), so a path that wanders off them there
      # (the map's sections in a junction) has the less room for it
      r_ = self.route_r
      lo_[i0:i1 + 1] = max(lo_[i0 - 1] + r_[i0 - 1], lo_[i1 + 1] + r_[i1 + 1]) - r_[i0:i1 + 1]
      hi_[i0:i1 + 1] = min(hi_[i0 - 1] + r_[i0 - 1], hi_[i1 + 1] + r_[i1 + 1]) - r_[i0:i1 + 1]
      self.room_tried[i0:i1 + 1] = True

  def _nudge(self, v: float, t: float):
    """Each parked vehicle (no one at its wheel, standing), or driven one standing across our way, newly in the way along
    the path ahead: the path shifted past it within our lane (_nudge_for, _nudge_add), else left for the lead to stop
    behind; a nudge event either way. A parked one our lane has no room past may be passed slowly through the room
    beside it (_borrow), looked at again every BORROW_RETRY s while that room is busy; an event as its outcome changes.
    Shifts and vehicles passed are dropped."""
    if not self.c["nudge"] or self.path0 is None:
      return
    for g in self.nudge_groups:
      if self.s > g["end"]:
        self.nudge_fixed = self.nudge_fixed + g["off"]  # behind the car: left as driven
    self.nudge_groups = [g for g in self.nudge_groups if self.s <= g["end"]]
    self.nudge_blocked = [b for b in self.nudge_blocked if self.s <= b["end"]]
    retry = [b for b in self.nudge_blocked if b.get("later") and t >= b["retry"]]
    self.nudge_blocked = [b for b in self.nudge_blocked if not any(b is r for r in retry)]
    for car in self.cars or []:
      if car.get("ped") or float(np.hypot(*car["vel"])) > LEAD_MOVING:
        continue
      c = car["c"].mean(axis=0)
      if car["driven"]:  # driven: passed only standing across our way, never one queued in our lane
        along, _ = self._place(c[None], LEAD_AHEAD + self.front + 2 * FRONT + 2.0, self.path0)
        j = int(np.clip(np.searchsorted(self.s_path, along[0]), 0, len(self.s_path) - 1)) if not np.isnan(along[0]) else None
        if j is None or not self._crosswise(car, float(self.theta0[j])):
          continue
      seen = [e["c"] for g in self.nudge_groups for e in g["entries"]] + [b["c"] for b in self.nudge_blocked]
      if any(float(np.hypot(*(x - c))) < NUDGE_SAME for x in seen):
        continue
      before = next((b for b in retry if float(np.hypot(*(b["c"] - c))) < NUDGE_SAME), None)
      e = self._nudge_for(car, v, t)
      if e is None:
        continue
      why = e.get("why") or self._nudge_try(e)
      if why is not None and e.get("shift") is not None and e.get("borrow") is None and car["driven"] is False and self.c["borrow"]:
        b = self._borrow(e, e["body"], v, t)  # a shift within our lane that doesn't fit with the others: wider
        if b.get("why") is None:
          e, why = b, self._nudge_try(b)
        else:
          e = {**e, "later": bool(b.get("later")), "past": self._past(b)}
          why = b["why"] if b.get("later") else why
      rec = {"ok": why is None, "pos": [round(float(c[0]), 1), round(float(c[1]), 1)], "at": e["at"], "right": e["right"],
             "need": e["need"], **({"driven": True} if car["driven"] else {}), **({"past": e["past"]} if e.get("past") else {})}
      if why is None:
        rec.update(shift=round(e["shift"], 2), clear=round(e["clear"], 2), ease=[round(e["L_in"], 1), round(e["L_out"], 1)],
                   v=round(e["v"], 1), v_ease=round(e["v_ease"], 1), **({"kind": e["kind"]} if e.get("kind") else {}))
      else:
        rec["why"] = why
        later = bool(e.get("later"))
        self.nudge_blocked.append({"c": c, "end": e["at"][1] + self.rear, "why": why, "later": later, "retry": t + BORROW_RETRY})
        if before is not None and before["why"] == why:
          continue  # looked at again, as it was: logged once
      self.nudge_log.append({"t": round(t, 3), **rec})
      self._event("nudge", t, **rec)

  def _nudge_try(self, e: dict) -> str | None:
    """_nudge_add, with a pass's room past our lane (_borrow) as well while it's tried; why it can't be, else None."""
    saved = None
    if e.get("borrow") is not None:
      ia, ib, lo_, hi_ = e["borrow"]
      saved = (self.borrow_lo.copy(), self.borrow_hi.copy())
      self.borrow_lo[ia:ib + 1] = np.fmin(self.borrow_lo[ia:ib + 1], lo_)
      self.borrow_hi[ia:ib + 1] = np.fmax(self.borrow_hi[ia:ib + 1], hi_)
    why = self._nudge_add(e)
    if why is not None and saved is not None:
      self.borrow_lo, self.borrow_hi = saved
      e["past"] = f"{why} (shift {e['shift']:.2f}, eases {e['L_in']:.1f}/{e['L_out']:.1f})"
    return why

  @staticmethod
  def _past(b: dict) -> str:
    """A failed _borrow's why, for the nudge event."""
    return b["why"] + (f" ({b['by']})" if b.get("by") else "")

  def _borrow_pending(self, car: dict) -> dict | None:
    """The blocked entry of a parked vehicle we may yet pass through the room beside it (_borrow, busy for now); None."""
    if car.get("driven") is not False:
      return None
    c = car["c"].mean(axis=0)
    return next((b for b in self.nudge_blocked if b.get("later") and float(np.hypot(*(b["c"] - c))) < NUDGE_SAME), None)

  def _ease_v(self, length: float, shift: float) -> float:
    """m/s an ease across `shift` m over `length` m is comfortable at."""
    a = NUDGE_A_LAT * self.style["a_lat"]
    return min(length / NUDGE_EASE_T, length * math.sqrt(a / (5.77 * max(abs(shift), 1e-3))))

  def _nudge_for(self, car: dict, v: float, t: float = 0.0) -> dict | None:
    """The shift past a parked vehicle: m right (`shift`, from `need` at the least), held over path m h0 to h1 (points i0
    to i1), eased in over L_in m and out over L_out (no faster than `v_ease` m/s), passed at `v` m/s at most with `clear`
    m between our bodies; with a `why` where there's none (and `later` where _borrow's way past may yet open). Where
    our lane has none for a parked one, _borrow's. None where it leaves us nudge_clear already, or isn't all in view
    ahead."""
    # placed on a little further than the lead looks, so it's all in view once it's a lead
    along, right = self._place(self._outline(car["c"]), LEAD_AHEAD + self.front + 2 * FRONT + 2.0, self.path0)
    if np.isnan(along).any():
      return None
    hw = self.half_width
    clear = max(float(self.c["nudge_clear"]), LEAD_SIDE + 0.05)
    want = max(float(self.c["nudge_want"]), clear)
    r0, r1, a0, a1 = float(right.min()), float(right.max()), float(along.min()), float(along.max())
    if r0 > hw + clear or r1 < -(hw + clear) or a0 < self.s + self.front - LEAD_BESIDE:
      return None  # clear of us, or beside us already
    sp = self.s_path
    h0, h1 = a0 - self.front - NUDGE_MARGIN, a1 + self.rear + NUDGE_MARGIN
    i0 = int(np.searchsorted(sp, h0))
    i1 = min(int(np.searchsorted(sp, h1, side="right")), len(sp) - 1)
    e = {"c": car["c"].mean(axis=0), "at": [round(a0, 1), round(a1, 1)], "right": [round(r0, 2), round(r1, 2)], "need": None,
         "body": (r0, r1, a0, a1)}

    def failed(why: str) -> dict:
      if car["driven"] is not False or not self.c["borrow"]:
        return {**e, "why": why}
      b = self._borrow(e, (r0, r1, a0, a1), v, t)
      if b.get("why") is None:
        return b
      return {**e, "why": b["why"] if b.get("later") else why, "later": bool(b.get("later")), "past": self._past(b)}
    self._fill_room(int(np.searchsorted(sp, h0 - NUDGE_FILL)), int(np.searchsorted(sp, h1 + NUDGE_FILL)))
    lo, hi = self.room_lo[i0:i1 + 1], self.room_hi[i0:i1 + 1]
    if not len(lo) or np.isnan(lo).any():
      return failed("no lane")
    options = []  # (shift, need): past it on its left (moved no further right than need), or on its right
    if r0 - hw - clear >= lo.max():
      options.append((max(r0 - hw - want, float(lo.max())), r0 - hw - clear))
    if r1 + hw + clear <= hi.min():
      options.append((min(r1 + hw + want, float(hi.min())), r1 + hw + clear))
    if not options:
      need = r0 - hw - clear if abs(r0 - hw - clear) < abs(r1 + hw + clear) else r1 + hw + clear
      e["need"] = round(need, 2)
      return failed("no room")
    shift, need = min(options, key=lambda o: abs(o[0]))
    e.update(need=round(need, 2), shift=shift, clear=clear + abs(shift - need), h0=h0, h1=h1, i0=i0, i1=i1)
    # eased in at the speed it's driven at there, slowing to pass it the tighter the pass
    v_ref = max(float(np.interp(h0, sp, self.v_static)), 1.0)
    a_lat = NUDGE_A_LAT * self.style["a_lat"]
    length = max(NUDGE_EASE_MIN, v_ref * NUDGE_EASE_T, v_ref * math.sqrt(5.77 * abs(shift) / a_lat))
    # eased in ahead of the car and out before the path leaves its lane's guide
    gaps = np.flatnonzero(np.isnan(self.room_lo[:i0]))
    lane0 = float(sp[gaps[-1] + 1] if len(gaps) else sp[0])
    gaps = np.flatnonzero(np.isnan(self.room_lo[i1 + 1:]))
    lane1 = float(sp[i1 + gaps[0]] if len(gaps) else sp[-1])
    e["L_in"] = min(length, h0 - max(self.s + NUDGE_GUARD, lane0))
    e["L_out"] = min(length, lane1 - h1)
    if e["L_in"] < NUDGE_EASE_MIN:
      return failed("late" if h0 - self.s - NUDGE_GUARD < NUDGE_EASE_MIN else "no lane")
    if e["L_out"] < NUDGE_EASE_MIN:
      return failed("no lane")
    e["v"] = NUDGE_V[0] + NUDGE_V[1] * (e["clear"] - clear)
    e["v_ease"] = min(self._ease_v(e["L_in"], shift), self._ease_v(e["L_out"], shift))
    d = h0 - e["L_in"] - self.s - v * STOP_LAG
    if math.sqrt(max(v * v - 2.0 * self.style["decel"] * max(d, 0.0), 0.0)) > NUDGE_LATE * e["v_ease"]:
      return failed("late")
    return e

  def _borrow(self, e: dict, body: tuple, v: float, t: float) -> dict:
    """Past a parked vehicle our lane alone has no room past (or none at the speed it's seen at): slowly, through the
    room beside our lane on its far side (_borrow_room), once that's clear of others for as long as the pass takes
    (_borrow_clear) and the car is within BORROW_RANGE of it, slow enough to ease out at BORROW_KAPPA. _nudge_for's
    entry, its `kind` the room's and `borrow` (path points a to b, their left and right bounds) for _nudge_bad; else
    {"why", "later": whether to look again (it's busy or not yet near, and _longitudinal stops BORROW_STOP short)}."""
    r0, r1, a0, a1 = body
    hw = self.half_width
    clear = max(float(self.c["nudge_clear"]), LEAD_SIDE + 0.05)
    near = -1 if abs(r0 - hw - clear) <= abs(r1 + hw + clear) else 1  # the side needing the smaller shift
    tries = [(side, self._borrow_side(e, body, v, t, side)) for side in (near, -near)]
    # either side whose room is there (passed now, or once clear): into a lane our way first, then over the line,
    # then the road's edge; the smaller shift where alike
    there = [(BORROW_PREFER.index(b["kind"]), k, b) for k, (side, b) in enumerate(tries) if b.get("kind") and (b.get("why") is None or b.get("later"))]
    return min(there, key=lambda x: (x[0], x[1]))[2] if there else tries[0][1]

  def _borrow_side(self, e: dict, body: tuple, v: float, t: float, side: int) -> dict:
    """_borrow past its left (`side` -1: the path moved left) or its right (1)."""
    r0, r1, a0, a1 = body
    hw = self.half_width
    clear = max(float(self.c["nudge_clear"]), LEAD_SIDE + 0.05)
    want = max(float(self.c["nudge_want"]), clear)
    sp, n = self.s_path, len(self.s_path)
    need = r0 - hw - clear if side < 0 else r1 + hw + clear
    h0, h1 = a0 - self.front - BORROW_MARGIN, a1 + self.rear + BORROW_MARGIN
    i0 = int(np.searchsorted(sp, h0))
    i1 = min(int(np.searchsorted(sp, h1, side="right")), n - 1)
    ia = max(int(np.searchsorted(sp, h0 - BORROW_EASE)), 0)
    ib = min(int(np.searchsorted(sp, h1 + BORROW_EASE)), n - 1)
    if i1 <= i0 or h0 <= self.s + NUDGE_GUARD:
      return {"why": "no room", "by": "near"}
    lo, hi, kinds = self._borrow_room(ia, ib, side)
    held_lo, held_hi = lo[i0 - ia:i1 - ia + 1], hi[i0 - ia:i1 - ia + 1]
    if np.isnan(held_lo).any():
      return {"why": "no lane"}
    wide = False
    if (held_lo.max() > need + 1e-6 if side < 0 else held_hi.min() < need - 1e-6) and "oncoming" in kinds[i0 - ia:i1 - ia + 1]:
      lo, hi, kinds = self._borrow_room(ia, ib, side, BORROW_WIDE)  # further over the line, the pass clear the wider
      held_lo, held_hi, wide = lo[i0 - ia:i1 - ia + 1], hi[i0 - ia:i1 - ia + 1], True
    if side < 0:
      bound = float(held_lo.max())
      if bound > need + 1e-6:
        return {"why": "no room", "by": f"{bound:.2f}"}
      shift = max(min(r0 - hw - want, need), bound)
    else:
      bound = float(held_hi.min())
      if bound < need - 1e-6:
        return {"why": "no room", "by": f"{bound:.2f}"}
      shift = min(max(r1 + hw + want, need), bound)
    kind = max(kinds[i0 - ia:i1 - ia + 1], key=BORROW_KINDS.index)  # the boldest the pass takes
    # eased in from the car (or where the room starts) and out to where it ends, no tighter than BORROW_KAPPA
    ok = ~np.isnan(lo)
    k0 = i0 - ia
    while k0 > 0 and ok[k0 - 1]:
      k0 -= 1
    k1 = i1 - ia
    while k1 < len(ok) - 1 and ok[k1 + 1]:
      k1 += 1
    L_in = min(BORROW_EASE, h0 - max(self.s + NUDGE_GUARD, float(sp[ia + k0])))
    L_out = min(BORROW_EASE, float(sp[ia + k1]) - h1)
    # where the eases are short (stopped near it, the room ending soon after), no further over than needed for them
    fits = min(L_in, L_out) ** 2 * BORROW_KAPPA / 5.77
    if min(L_in, L_out) < BORROW_EASE_MIN or fits < abs(need):
      return {"why": "no room", "by": f"eases {L_in:.1f}/{L_out:.1f} for {abs(need):.2f}"}
    shift = math.copysign(min(abs(shift), fits), shift)
    if a0 - self.s - self.front > BORROW_RANGE:
      return {"why": "wait", "later": True, "kind": kind}  # decided once near, on what's about then
    v_ease = min(BORROW_V, self._ease_v(L_in, shift), self._ease_v(L_out, shift))
    d = h0 - L_in - self.s - v * STOP_LAG
    if math.sqrt(max(v * v - 2.0 * self.style["decel"] * max(d, 0.0), 0.0)) > NUDGE_LATE * v_ease:
      return {"why": "late", "later": True, "kind": kind}
    busy = self._borrow_clear(e["c"], side, shift, h1 + L_out, kind, t, wide)
    if busy is None and kind == "lane" and not self._lane_clear("left" if side < 0 else "right", self.last_state):
      busy = "lane"  # the lane beside, by the lane change's own check
    if busy is not None:
      return {"why": "busy", "later": True, "by": busy, "kind": kind}
    return {**e, "need": round(need, 2), "shift": shift, "clear": clear + abs(shift - need), "h0": h0, "h1": h1, "i0": i0,
            "i1": i1, "L_in": L_in, "L_out": L_out, "v": BORROW_V, "v_ease": v_ease, "kind": kind, "borrow": (ia, ib, lo, hi)}

  def _borrow_room(self, a: int, b: int, side: int, over: float = BORROW_ONCOMING) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """How far (m right of path0) path points a to b may move past a parked vehicle on its `side` (-1 left), by the
    road's sections: our body BORROW_LANE into the next lane our way ("lane"), `over` m over the line into the
    oncoming or a centre lane ("oncoming"), or BORROW_EDGE in from the road's edge ("edge"); within our lane's
    CLAMP_MARGIN on the other side. (left, right bounds, NaN where the sections are no guide, and kinds): about turns
    and where the path leaves its road's line, as _fill_room's, but not about junction nodes the path runs straight
    through (parked cars stand by side turnings and driveways), a lane there of NUDGE_SPAN's width."""
    route, lanes = self.route, self.route.lanes
    turns = np.asarray(self._turn_alongs, float)
    hw = self.half_width
    lo, hi, kinds = np.full(b - a + 1, np.nan), np.full(b - a + 1, np.nan), [""] * (b - a + 1)
    for i in range(a, b + 1):
      rs, r = float(self.route_s[i]), float(self.route_r[i])
      if len(turns) and (np.abs(turns - rs) < NUDGE_NODE).any():
        continue
      k = segment_of(route, rs)
      d = route.points[k + 1, :2] - route.points[k, :2]
      sec = lanes.section_at(rs, k)
      if sec is None or not sec.lanes:
        continue
      if abs(wrap(math.degrees(self.theta0[i] - math.atan2(d[1], d[0])))) > 2.0 * NUDGE_ALIGN:  # a lane's swerve as it narrows
        continue
      j = next((j for j in range(sec.first, sec.first + sec.lanes) if sec.spans[j].left - 0.05 <= r <= sec.spans[j].right + 0.05), None)
      if j is None:
        continue
      own = sec.spans[j]
      if not NUDGE_SPAN[0] <= own.right - own.left <= NUDGE_SPAN[1]:
        continue
      near = max(own.left, sec.edges[0]) + hw + CLAMP_MARGIN - r, min(own.right, sec.edges[1]) - hw - CLAMP_MARGIN - r
      nb = j + side
      if 0 <= nb < len(sec.spans) and abs((own.left - sec.spans[nb].right) if side < 0 else (sec.spans[nb].left - own.right)) > 0.3:
        kind, edge = "edge", (own.left + BORROW_EDGE) if side < 0 else (own.right - BORROW_EDGE)  # a median between
      elif not 0 <= nb < len(sec.spans):  # our lane's own edge, not a shoulder past it (a freeway's barrier stands there)
        kind, edge = "edge", (max(own.left, sec.edges[0]) + BORROW_EDGE) if side < 0 else (min(own.right, sec.edges[1]) - BORROW_EDGE)
      elif sec.spans[nb].heading == 1:
        far = sec.spans[nb].left + CLAMP_MARGIN if side < 0 else sec.spans[nb].right - CLAMP_MARGIN
        kind, edge = "lane", max(own.left - BORROW_LANE, far) if side < 0 else min(own.right + BORROW_LANE, far)
      else:
        kind, edge = "oncoming", own.left - over if side < 0 else own.right + over
      # the path as planned is room, whatever the sections say (the kerb side's bound only matters past it)
      if side < 0:
        lo[i - a], hi[i - a] = min(edge + hw - r, near[0]), max(near[1], 0.0)
      else:
        lo[i - a], hi[i - a] = min(near[0], 0.0), max(edge - hw - r, near[1])
      kinds[i - a] = kind
    return lo, hi, kinds

  def _borrow_clear(self, blocker: np.ndarray, side: int, shift: float, end: float, kind: str, t: float,
                    wide: bool = False) -> str | None:
    """Why the room a pass past a parked vehicle (middle `blocker`) takes isn't clear, else None: anyone else (on foot
    too) in it now, or coming into it straight on at their speed before the car's back in its lane (path m `end`, at
    BORROW_V from where it is, BORROW_SPARE s more): from BORROW_BEHIND m behind our rear (only the room beyond our body
    there) to BORROW_AHEAD past `end`; BORROW_SIDE m more beside it, more still into the oncoming lanes (and twice that,
    and twice BORROW_SPARE, for a `wide` pass). Other parked ones on the blocker's side of our way are left to their own
    nudges."""
    hw = self.half_width
    pad = BORROW_SIDE + (BORROW_SIDE_ONCOMING if kind == "oncoming" else 0.0) * (2.0 if wide else 1.0)
    band = (shift - hw - pad, shift + hw + pad)
    beyond = (band[0], -hw - 0.2) if side < 0 else (hw + 0.2, band[1])  # beside and behind us: past our own body only
    secs = (end - self.s) / BORROW_V + BORROW_SPARE * (2.0 if wide else 1.0)
    ahead = end + BORROW_AHEAD - self.s
    back = self.rear + BORROW_BEHIND
    for car in self.cars or []:
      c = car["c"].mean(axis=0)
      if float(np.hypot(*(c - blocker))) < NUDGE_SAME:
        continue
      moving = float(np.hypot(*car["vel"])) >= LEAD_MOVING
      for dt in np.arange(0.0, secs + 0.01, 0.5) if moving else (0.0,):
        along, right = self._place(self._outline(self._car_now(car, t, dt)), ahead, self.path0, behind=back)
        ok = ~np.isnan(along)
        if not moving and car.get("driven") is False and ok.any() and side * float(np.nanmean(right)) < 0.0:
          break
        front = ok & (along >= self.s + self.front)
        inside = (front & (right > band[0]) & (right < band[1])) | (ok & ~front & (right > beyond[0]) & (right < beyond[1]))
        if inside.any():
          return "ped" if car.get("ped") else "coming" if moving else "standing"
    return None

  def _group(self, entries: list[dict]) -> dict:
    """Shifts one way whose eases overlap, as one: each held over its vehicle, the gap between two held at the lesser,
    each step up eased in before where it's held (its L_in), each down after (L_out): {"off": m right at each path
    point, "start", "end": path m it's under way over, "v": m/s its eases allow over it, "sign", "entries"}."""
    sp = self.s_path
    es = sorted(entries, key=lambda e: e["h0"])
    sign = math.copysign(1.0, es[0]["shift"])
    lev = np.zeros(len(sp))
    for e in es:
      lev[e["i0"]:e["i1"] + 1] = np.maximum(lev[e["i0"]:e["i1"] + 1], abs(e["shift"]))
    reach, last = es[0]["i1"], es[0]
    for e in es[1:]:
      if e["i0"] > reach + 1:
        lev[reach + 1:e["i0"]] = min(abs(last["shift"]), abs(e["shift"]))
      if e["i1"] > reach:
        reach, last = e["i1"], e
    ins, outs = {e["i0"]: e for e in es}, {e["i1"]: e for e in es}
    off = np.zeros(len(sp))
    start, end, v = math.inf, -math.inf, min(e["v_ease"] for e in es)
    for j in np.flatnonzero(np.diff(lev)):
      a, b = lev[j], lev[j + 1]
      if b > a:
        length = ins[j + 1]["L_in"] if j + 1 in ins else min(e["L_in"] for e in es)
        s0 = sp[j + 1] - length
      else:
        length = outs[j]["L_out"] if j in outs else min(e["L_out"] for e in es)
        s0 = sp[j]
      off += (b - a) * quintic((sp - s0) / length)
      start, end = min(start, s0), max(end, s0 + length)
      v = min(v, self._ease_v(length, b - a))
    return {"off": sign * off, "start": float(start), "end": float(end), "v": float(v), "sign": sign, "entries": es}

  def _nudge_add(self, e: dict) -> str | None:
    """A shift into the plan: with the shifts one way it overlaps, else on its own; None, or why it can't be."""
    sign = math.copysign(1.0, e["shift"])
    e0, e1 = e["h0"] - e["L_in"], e["h1"] + e["L_out"]
    tries = [self.nudge_groups[:k] + [self._group(g["entries"] + [e])] + self.nudge_groups[k + 1:]
             for k, g in enumerate(self.nudge_groups) if g["sign"] == sign and e0 < g["end"] and g["start"] < e1]
    tries.append(self.nudge_groups + [self._group([e])])
    why = None
    for groups in tries:
      off = self.nudge_fixed + np.sum([g["off"] for g in groups], axis=0)
      why = self._nudge_bad(groups, off)
      if why is None:
        self.nudge_groups = groups
        self._nudge_apply(off)
        return None
    return why

  def _nudge_bad(self, groups: list[dict], off: np.ndarray) -> str | None:
    """Why the shifts `off` can't be driven: the path moved where the car is, outside the lane's room (and a pass's
    room past it, _borrow's) or where it's no guide, or short of a vehicle's need; else None."""
    sp = self.s_path
    near = sp <= self.s + NUDGE_GUARD
    if np.abs(off[near] - self.nudge_off[near]).max(initial=0.0) > 0.01:
      return "under way"
    moved = np.abs(off) > 0.02
    lo, hi = np.fmin(self.room_lo, self.borrow_lo), np.fmax(self.room_hi, self.borrow_hi)
    if (moved & np.isnan(lo)).any():
      return "no lane"
    m = moved & ~np.isnan(lo)
    if (off[m] < lo[m] - 0.01).any() or (off[m] > hi[m] + 0.01).any():
      return "no room"
    for g in groups:
      for e in g["entries"]:
        held = off[e["i0"]:e["i1"] + 1]
        if (held.max() > e["need"] + 0.01) if e["shift"] < 0 else (held.min() < e["need"] - 0.01):
          return "conflict"
    return None

  def _nudge_apply(self, off: np.ndarray):
    """The path moved `off` m right of path0, and its heading and curvature anew (its length along kept: the moves are
    small and gentle)."""
    self.nudge_off = off
    th = headings(self.path0)
    self.path = self.path0 + np.stack([np.sin(th), -np.cos(th)], axis=1) * off[:, None]
    self.theta = headings(self.path)
    self.kappa_path = smooth(np.gradient(self.theta, self.s_path), max(int(CURV_SMOOTH / STEP) | 1, 1))

  # *** each step ***

  def step(self, route, state: dict, t: float, collisions: int = 0) -> dict | None:
    """The plugin control message for this bridge step (t: s, a monotonic clock); None until there's a route."""
    if self.finished is not None:
      return {"type": "control", "active": True, "curvature": 0.0, "accel": HOLD_ACCEL} if self.finished == "arrived" else None
    pos = state.get("pos")
    v = float(state.get("vEgo") or 0.0)
    if pos is None:
      return self.msg
    self._body(state)
    key =(round(pos[0], 3), round(pos[1], 3), round(float(state.get("heading") or 0.0), 3), round(v, 3))
    if key == self.last_key and self.msg is not None and self.last_t is not None and t - self.last_t < COMPUTE_EVERY:
      return self.msg  # the same game frame: the control again, as the plugin drops one older than 0.3 s
    dt = min(max(t - self.last_t, 0.02), 0.2) if self.last_t is not None else 0.05
    self.last_key, self.last_t = key, t
    if self.t0 is None:
      self.t0, self.collisions0, self.engage0 = t, collisions, state.get("engagePresses")
      self.progress = (0.0, t)
    if route is None:
      self.progress = (0.0, t)  # waiting for the router isn't being stuck
      self._set("wait", t)
      self.msg = {"type": "control", "active": True, "curvature": 0.0, "accel": HOLD_ACCEL}
      return self.msg
    if route is not self.route and self.phase != "abort":
      if self.path is None:
        if not self._make_plan(route, state, t):
          return self._abort_msg(v, t)
        self._set("drive", t, plan=self.plan, style=self.style, seed=self.seed, version=map_version())
      else:
        self._splice(route, state, t)
    if self.phase == "abort":
      return self._abort_msg(v, t)
    self._vehicles(state, t)
    self._obstacles(state, t)
    self.last_v = v
    self.last_state = state
    if self.asked is not None:
      self._ask_step(route, state, t)
    self._lc_gate(route, state, t, v)
    if self.phase == "abort":
      return self._abort_msg(v, t)
    self._checks(route, state, t, collisions, v)
    if self.phase == "abort":
      return self._abort_msg(v, t)
    if self.reversing is not None:
      msg = self._reverse(v, t, dt)
      if msg is not None:
        self.msg = msg
        return msg
    self._nudge(v, t)
    accel = self._longitudinal(state, v, t, dt)
    if self.phase == "abort":
      return self._abort_msg(v, t)
    curvature = self._lateral(state, v, t, dt)
    self._labels(route, state, v, t)
    self._watch(route, state, v, t)
    self.msg = {"type": "control", "active": True, "curvature": round(curvature, 5), "accel": round(accel, 3)}
    return self.msg

  def _abort_msg(self, v: float, t: float) -> dict:
    self.indicator, self.label = None, "none"
    self.a = HOLD_ACCEL if v < STOPPED else -ABORT_DECEL
    self.kappa = 0.0
    self.reason = "abort"
    if v < STOPPED and self.finished is None:
      self.finished = f"abort: {self.why}"
      self._event("mapdrive", t, phase="end", finished=self.finished)
    self.msg = {"type": "control", "active": True, "curvature": 0.0, "accel": self.a}
    return self.msg

  def _checks(self, route, state: dict, t: float, collisions: int, v: float):
    pos = state["pos"]
    self.s, self.dev = project(self.path, self.s_path, pos, self.s)

    def held(name: str, cond: bool, secs: float) -> bool:
      start = self.timers.get(name)
      start = (start if start is not None else t) if cond else None
      self.timers[name] = start
      return start is not None and t - start >= secs
    user = state.get("user") or {}
    engage = state.get("engagePresses")
    if collisions > (self.collisions0 or 0):
      self._abort("collision", t, pos)
    elif abs(float(user.get("steer") or 0.0)) > USER_STEER or user.get("brake") or user.get("gas"):
      self._abort("driver input", t, pos)
    elif engage is not None and self.engage0 is not None and engage > self.engage0:
      self._abort("the driver pressed the engage key", t, pos)
    elif held("dev", self.dev > DEV_ABORT, DEV_ABORT_S):
      self._abort(f"{self.dev:.1f} m off the path", t, pos)
    herr = abs(wrap(float(state.get("heading") or 0.0) - game_heading(float(np.interp(self.s, self.s_path, self.theta)))))
    if self.phase != "abort" and held("heading", herr > HEADING_ABORT and v > 2.0, HEADING_ABORT_S):
      self._abort(f"heading {herr:.0f} deg off the path", t, pos)
    a_lat = abs(v * float(state.get("yawRate") or 0.0))
    if self.phase != "abort" and held("a_lat", a_lat > A_LAT_MAX, A_LAT_ABORT_S):
      self._abort(f"lateral acceleration {a_lat:.1f} m/s^2", t, pos)
    # the lane matcher's oncoming lanes away from a junction: a plan gone wrong, whatever its keys say
    oncoming = (state.get("laneMap") or {}).get("kind") == "oncoming" and not self._unclear(route.at)
    # nor while passing a parked car over the line (_borrow)
    oncoming = oncoming and not any(g["start"] <= self.s <= g["end"] and any(e.get("kind") == "oncoming" for e in g["entries"])
                                    for g in self.nudge_groups)
    if self.phase != "abort" and held("oncoming", oncoming, ONCOMING_S):
      self._abort("in the oncoming lanes", t, pos)
    if self.s > self.progress[0] + 1.0:
      self.progress = (self.s, t)
    waiting = self.phase in ("stop", "arrive", "follow", "yield") or self.stopped_at is not None
    if waiting:
      self.progress = (self.progress[0], t)
    if self.phase != "abort" and t - self.progress[1] > NO_PROGRESS_S:
      self._abort(f"no progress in {NO_PROGRESS_S:.0f} s", t, pos)
    if self.phase != "abort" and route.off < 30.0 and not route.elsewhere:
      sec = route.section(route.seg)
      near_j = self._unclear(route.at) or self.s < self.join_m
      # off the road and off the plan: on the plan, it's the plan past the map's kerbs (a kerb_contact anomaly), not
      # the driving
      if sec is not None and sec.lanes and not near_j and self.dev > OFF_ROAD_DEV and \
         not (sec.edges[0] - OFF_ROAD <= route.right <= sec.edges[1] + OFF_ROAD):
        if held("off_road", True, 0.5):
          self._abort(f"off the road ({route.right:+.1f} m right of its line)", t, pos)
      else:
        self.timers["off_road"] = None

  def _longitudinal(self, state: dict, v: float, t: float, dt: float) -> float:
    st = self.style
    s = self.s
    sp = self.s_path
    self.lead_gap, self.lead_right, self.lead_driven = math.nan, math.nan, None
    look = max(2.0, 1.2 * v)
    vp_here = float(np.interp(s, sp, self.v_static))
    vp = float(np.interp(s + look, sp, self.v_static))
    reason = REASONS[int(self.why_static[min(int((s + look) / STEP), len(sp) - 1)])]
    phase = "drive"
    b = st["decel"]
    stop_d = None  # m on to the stop the car is to come to next
    # the next stop sign or give way line not yet done
    for x in self.stops:
      if x["done"]:
        continue
      if x["kind"] == "lights":
        light = self._light(x, state, t)
        if s + self.front >= x["s"]:
          x["done"] = True
          mark = {"t": round(t, 3), "s": round(x["s"], 1), "along": round(x["along"], 1), "from": round(t - LIGHT_MARK, 3), "to": round(t + LIGHT_MARK, 3)}
          self.lights.append(mark)
          self._event("light_unknown", t, **{k: v for k, v in mark.items() if k != "t"}, **({"seen": light} if light else {}))
          continue
        d = x["target"] - s
        if light == "red" and (x.get("held") is not None or d + 1.0 >= v * STOP_LAG + v * v / (2 * DECEL_REAL)):
          # the traffic shows it red: stopped at its line, as for a stop sign, and held there while it does
          if x.get("held") is None:
            x["held"] = t
            self._event("light_stop", t, why=x.get("why"), d=round(d, 1))
          if t - x["held"] > LIGHT_WAIT_MAX:
            self._abort(f"held {LIGHT_WAIT_MAX:.0f} s at a red light", t, state.get("pos"))
            return self.a
          cap = math.sqrt(2 * b * 0.8 * max(d - 0.3, 0.0))
          stop_d = d
          self.stopped_at = t if v < STOPPED else None
          if cap < vp:
            vp, reason = cap, "light"
            if d < 40.0:
              phase = "stop"
          break
        if x.get("held") is not None:
          self._event("light_go", t, waited=round(t - x["held"], 1), why=x.get("why"))
          x["held"] = None
          self.stopped_at = None
        if light is None and x.get("traffic") and 0.0 < x["s"] - s - self.front < LIGHT_LOOK:
          # traffic about but nothing shown of the light yet: no faster than would stop at its line from LIGHT_UNKNOWN_V
          cap = math.sqrt(LIGHT_UNKNOWN_V ** 2 + 2 * b * 0.8 * max(d - look, 0.0))
          if cap < vp:
            vp, reason = cap, "light"
        continue
      d = x["target"] - s
      if x["kind"] == "give_way":
        if s + self.front >= x["s"]:
          x["done"] = True
          continue
        cap = math.sqrt((GIVE_WAY_SPEED * st["speed"]) ** 2 + 2 * b * 0.8 * max(d, 0.0))
        if cap < vp:
          vp, reason = cap, "give_way"
          if d < 30.0:
            phase = "give_way"
        break
      # a stop sign: a full stop at its target, the dwell, then on
      if x["t_stop"] is not None:
        if t - x["t_stop"] >= st["stop_dwell"] + st["reaction"]:
          first = self._before_us(x, t) if t - x["t_stop"] < st["stop_dwell"] + st["reaction"] + ORDER_WAIT else None
          if first is None:
            if x.get("order"):
              self._event("stop_order", t, waited=round(t - x["t_stop"], 1), gave_way=x["order"])
            x["done"] = True
            self.stopped_at = None
            continue
          x["order"] = x.get("order", 0) + (first["id"] != x.get("first"))
          x["first"] = first["id"]
          self.stopped_at = x["t_stop"]
          self.v_prof, self.reason = 0.0, "order"
          self._set("stop", t)
          return self._out_accel(HOLD_ACCEL, dt, hard=True)
        self.stopped_at = x["t_stop"]
        self.v_prof, self.reason = 0.0, "stop"
        self._set("stop", t)
        return self._out_accel(HOLD_ACCEL, dt, hard=True)
      if (d < STOP_NEAR and v < STOPPED) or d < -1.0:
        x["t_stop"] = t
        self.stopped_at = t
        self._event("stopped", t, kind="stop", miss=round(d, 2))
        self.v_prof, self.reason = 0.0, "stop"
        self._set("stop", t)
        return self._out_accel(HOLD_ACCEL, dt, hard=True)
      cap = math.sqrt(2 * b * 0.8 * max(d - 0.3, 0.0))
      stop_d = d
      if cap < vp:
        vp, reason = cap, "stop"
        if d < 40.0:
          phase = "stop"
      break
    # the arrival
    d_end = self.s_arrive - s
    stop_d = min(stop_d, d_end) if stop_d is not None else d_end
    if d_end < STOP_NEAR and v < STOPPED or d_end < -1.0:
      if self.arrived_t is None:
        self.arrived_t = t
        self._event("arrived", t, miss=round(d_end, 2))
      self._set("arrive", t)
      self.v_prof, self.reason = 0.0, "arrive"
      if t - self.arrived_t >= float(self.c["hold_after"]) and self.finished is None:
        self.finished = "arrived"
        self._event("mapdrive", t, phase="end", finished="arrived")
      return self._out_accel(HOLD_ACCEL, dt, hard=True)
    # passing a parked vehicle: no faster than its shift's eases are comfortable at, nor its pass
    for g in self.nudge_groups:
      if s <= g["end"]:
        for at, cap in [(g["start"], g["v"])] + [(e["h0"], e["v"]) for e in g["entries"] if s <= e["h1"]]:
          cap = math.sqrt(cap * cap + 2 * b * 0.8 * max(at - s - look, 0.0))  # vp is the speed look m on
          if cap < vp:
            vp, reason = cap, "nudge"
    # vehicles: no faster than stops for one at the edge of what the plugin reports (but for range_share of the speed),
    # one about to cross our way, and the one in it
    detail: dict = {}
    holder: dict = {}  # the vehicle followed, holding the car up
    need = math.inf  # the acceleration a vehicle holding the car up allows
    if self.cars is not None:
      vis = self._visible()
      k = DECEL_REAL * RANGE_LAG
      cap = max(-k + math.sqrt(k * k + 2 * DECEL_REAL * max(vis - RANGE_STOP, 0.0)), float(self.c["range_share"]) * vp)
      if cap < vp:
        vp, reason = cap, "range"
      cross = self._crossing(t, v)
      if cross is not None and not cross[1].get("ped"):
        line = self._line_short_of(cross[1]["at"], v)  # held at the junction's line, not in the lanes before the crosser's
        if line is not None and line < cross[0]:
          cross = (line, {**cross[1], "line": True})
      held_in = v < STOPPED and self._in_junction()  # stopped in a junction: held on longer, and where we are
      keep = CROSS_KEEP_IN if held_in else CROSS_KEEP
      if cross is not None:
        self.cross_hold = (s + self.front + cross[0], t, cross[1])
      elif self.cross_hold is not None and t - self.cross_hold[1] < keep and \
           s + self.front < self.cross_hold[0] + (CROSS_STOP + 1.0 if held_in else 0.0):
        cross = (self.cross_hold[0] - s - self.front, self.cross_hold[2])  # held on through a moment's doubt
      if cross is not None:
        d, info = cross
        if held_in:  # stopped in a junction: not crept on towards the crosser's lane, each hold a little further in
          d = min(d, 0.0)
        cap = math.sqrt(2 * b * 0.8 * max(d - 0.3, 0.0))
        stop_d = min(stop_d, d)
        if cap < vp:
          vp, reason, phase, detail = cap, "ped" if info.get("ped") else "cross", "yield", info
          if v > cap:  # as much as stops short of it, no harder where it was seen late
            need = -v * v / (2.0 * max(d - v * STOP_LAG, 0.15))
      kerb = self._ped_kerb(t, v, look)
      if kerb is not None and kerb[0] < vp:
        vp, reason = kerb[0], "ped_kerb"
        if t - self.kerb_logged > 2.0:  # once as it starts holding the car back, not each time it takes over again
          self._event("ped_kerb", t, **kerb[1])
        self.kerb_logged = t
      lead = self._lead(t)
      if lead is not None:
        gap, vl, car, side = lead
        self.lead_gap, self.lead_right, self.lead_driven = gap, side, car["driven"]
        # a parked one we may yet pass through the room beside it: stopped BORROW_STOP short, room to pull out
        a_lead = self._follow(gap - (BORROW_STOP - LEAD_STOP) if self._borrow_pending(car) else gap, vl, v)
        if held_in and car.get("across_way"):
          a_lead = min(a_lead, HOLD_ACCEL)  # likewise for one across our way ahead: held where we are
        if a_lead < min(need, (vp * vp - v * v) / (2.0 * look)):
          cap = math.sqrt(vl * vl + 2 * b * max(gap - LEAD_STOP - st["headway"] * vl, 0.0))  # what it holds us to, for the record
          across = bool(car.get("crosswise"))  # standing across our way: waited for while it moves, not queued behind
          vp, reason, phase, need = (min(cap, vp), "ped" if car.get("ped") else "obstacle" if car.get("obst") else "blocked" if across else "lead",
                                     "yield" if across else "follow", a_lead)
          detail = {"gap": round(gap, 1), "speed": round(vl, 1), "right": round(side, 2), "driven": car["driven"],
                    **({"ped": True} if car.get("ped") else {}), **({"across": True} if across else {}),
                    **({"obst": True} if car.get("obst") else {})}
          holder = car
      # a plugin whose nearby has no pedestrians still counts those just ahead (traffic.peds, PED_BOX): stop for them
      peds = int((state.get("traffic") or {}).get("peds") or 0)
      if not self.peds_reported and peds:
        vp, reason, phase, detail = 0.0, "ped", "yield", {"peds": peds}
        if v > STOPPED:
          need = min(need, -DECEL_MAX)
    # a lane change waiting for its lane (_lc_gate): one keeping alongside in it let draw ahead, by dropping back (as
    # one closing from behind is let by at our speed); one that can't be put off any more, stopped short of
    lw = self.lc_wait
    if lw is not None and not lw["late"] and lw.get("why") == "beside":
      cap = max(LC_DROP_BACK * lw["v0"], 3.0)
      if cap < vp:
        vp, reason = cap, "hold"
    if lw is not None and lw["late"]:
      d = lw["s0"] - 1.0 - s
      cap = math.sqrt(2 * b * 0.8 * max(d - 0.3, 0.0))
      stop_d = min(stop_d, d)
      if cap < vp:
        vp, reason, phase, detail = cap, "hold", "yield", {"lane_gap": lw["side"]}
    # held up: the wait starts again whenever the vehicle holding us moves (a queue at a light we can't see creeping on);
    # one that's parked won't, so it ends the trip sooner
    held = phase in ("follow", "yield") and v < STOPPED and not (phase == "follow" and float(np.hypot(*holder["vel"])) > LEAD_MOVING)
    self.held_since = (self.held_since if self.held_since is not None else t) if held else None
    parked = phase == "follow" and holder["driven"] is False
    # one standing across our way may be waiting for us to clear its way, which we can't (no reverse): a deadlock
    blocked = phase == "yield" and (bool(holder.get("crosswise")) or bool(detail.get("slowed")))
    obst = phase == "follow" and bool(holder.get("obst"))  # a static thing on the plan: it won't move
    wait = (BORROW_WAIT if self._borrow_pending(holder) else PARKED_WAIT) if parked else DEADLOCK_WAIT if blocked else \
      OBST_WAIT if obst else LIGHT_WAIT_MAX if self._red_ahead(t) else WAIT_MAX  # in a queue at a light shown red: as long as it holds us
    if detail.get("lane_gap"):
      wait = LC_LATE_WAIT + 1.0  # stopped short of a change waiting for its lane: the gate ends it (_lc_gate)
    # one standing across our way close ahead, it may be waiting for us to clear its way: backed off a little where the
    # road behind is clear, to let it by (_reverse)
    near = detail.get("gap", detail.get("at", math.inf) - s - self.front if "at" in detail else math.inf)
    if (self.held_since is not None and t - self.held_since > REVERSE_AFTER and phase in ("follow", "yield") and blocked and
        near < REVERSE_NEAR and self.reverses < REVERSE_MAX and s > REVERSE_DIST + 2.0 and self._behind_clear(t)):
      self.reversing = {"t0": t, "to": s - REVERSE_DIST, "stopped": None}
      self.reverses += 1
      self.held_since = None
      self._event("reverse", t, gap=round(near, 1), n=self.reverses)
    if self.held_since is not None and t - self.held_since > wait:
      who = "pedestrian" if reason == "ped" else "parked vehicle" if parked else "vehicle standing across our way" if blocked else \
            "static obstacle" if obst else "vehicle"
      self._abort(f"held {wait:.0f} s by a {who}" + (" (deadlock)" if blocked else " in the way"), t, state.get("pos"))
      return self.a
    self._set(phase, t, **detail)
    self.v_prof, self.reason = min(vp_here, vp), reason
    if v < 0.5 and vp > 0.5 and reason not in ("stop", "arrive"):
      self.reason = "start"
    a = (vp * vp - v * v) / (2.0 * look)
    if not math.isinf(need):
      a = need
    # the last metres to a stop: the deceleration that stops at it, from where the car is once the controls act
    d_eff = stop_d - 0.2 - v * STOP_LAG
    a_stop = -v * v / (2.0 * max(d_eff, 0.15))
    if a_stop < min(a, -0.2) and d_eff < 30.0:
      a = a_stop
    if vp < 0.3 and v < 1.0:
      a = min(a, -0.5) if (self.s_arrive - s) > 0 else HOLD_ACCEL
    a = float(np.clip(a, -DECEL_MAX, st["accel"]))
    return self._out_accel(a, dt, hard=a < -b)

  def _behind_clear(self, t: float) -> bool:
    """Whether the road behind us is clear to back off REVERSE_DIST: no one (vehicle, pedestrian or static thing)
    within our width and a metre of the path from our rear bumper to REVERSE_BEHIND m past where the back-off ends, now
    or (by _predict) over the next REVERSE_T s."""
    back = REVERSE_DIST + REVERSE_BEHIND
    lo, hi = self.s - self.rear - back, self.s - self.rear + 0.5
    times = np.arange(0.0, REVERSE_T + 1e-6, 1.0)
    for car in (self.cars or []) + self.obst:
      corners = self._predict(car, t, times)
      along, right = self._place(np.concatenate([self._outline(c) for c in corners]), 1.0, behind=back + self.rear + 5.0)
      hit = ~np.isnan(along) & (along > lo) & (along < hi) & (np.abs(np.nan_to_num(right, nan=99.0)) < self.half_width + 1.0)
      if hit.any():
        return False
    return True

  def _reverse(self, v: float, t: float, dt: float) -> dict | None:
    """Backing off a deadlock: REVERSE_V straight back until REVERSE_DIST is done (or REVERSE_T s, or someone comes
    into the road behind), then standing still REVERSE_WAIT s for the one in our way to go by; None once that's done."""
    rv = self.reversing
    self.v_prof, self.reason, self.kappa = 0.0, "hold", 0.0
    if rv["stopped"] is None and (self.s <= rv["to"] or t - rv["t0"] > REVERSE_T or not self._behind_clear(t)):
      rv["stopped"] = t
      self._event("reversed", t, back=round(rv["to"] + REVERSE_DIST - self.s, 1))
    if rv["stopped"] is None:
      self._set("yield", t, reverse=True)
      self.a = -0.5
      return {"type": "control", "active": True, "curvature": 0.0, "accel": self.a, "reverse": REVERSE_V}
    if t - rv["stopped"] < REVERSE_WAIT or v < -STOPPED:
      self._set("yield", t, reversed=True)
      self.a = HOLD_ACCEL
      return {"type": "control", "active": True, "curvature": 0.0, "accel": self.a}
    self.reversing = None
    return None

  def _follow(self, gap: float, vl: float, v: float) -> float:
    """Our acceleration behind a vehicle gap m ahead going vl m/s along the path (the intelligent driver model's
    braking term, by the style's accel, decel and headway), no harder than the style's decel unless stopping LEAD_STOP
    short of it (as it slows to its speed) takes more: then just that."""
    st = self.style
    acc, b, hw = st["accel"], st["decel"], st["headway"]
    want = LEAD_STOP + v * hw + v * (v - vl) / (2.0 * math.sqrt(acc * b))
    a = acc * (1.0 - (max(want, 0.0) / max(gap, 0.1)) ** 2)
    if v <= vl:
      return a
    need = -(v * v - vl * vl) / (2.0 * max(gap - LEAD_STOP - v * STOP_LAG, 0.15))
    return need if need < -b else max(a, -b)

  def _out_accel(self, a: float, dt: float, hard: bool = False) -> float:
    j = self.style["jerk"] * (3.0 if hard else 1.0)
    if a < self.a:
      a = max(a, self.a - 2.5 * j * dt) if not hard else max(a, self.a - 4.0 * j * dt)
    else:
      a = min(a, self.a + j * dt)
    self.a = float(a)
    return self.a

  def _lateral(self, state: dict, v: float, t: float, dt: float) -> float:
    pos = np.asarray(state["pos"][:2], float)
    h = float(state.get("heading") or 0.0)
    yaw = float(state.get("yawRate") or 0.0)
    # steered from the rear axle, which moves along the car's heading: the game's position is the car's middle, which in
    # a bend moves inwards of its heading (b x curvature), and steering from it cuts every corner by about b x
    # curvature x lookahead (0.7 m in a square turn)
    back = float(self.c["rear_axle"]) if self.c.get("rear_axle") is not None else float(state.get("wheelBase") or WHEEL_BASE) / 2
    hr = math.radians(h)
    pos = pos - back * np.array([-math.sin(hr), math.cos(hr)])
    # the pose when the control acts
    lat = float(self.c["latency"])
    hm = math.radians(h + math.degrees(yaw * lat) / 2)
    pred = pos + v * lat * np.array([-math.sin(hm), math.cos(hm)])
    h_pred = h + math.degrees(yaw * lat)
    s_pred, _ = project(self.path, self.s_path, pred, self.s, window=40.0)
    ld = min(max(LOOKAHEAD[0], LOOKAHEAD[1] * v + LOOKAHEAD[2]), LOOKAHEAD[3])
    sp = self.s_path
    k_ff = float(np.interp(s_pred + v * float(self.c["ff_preview"]), sp, self.kappa_path))
    k_pp = pursuit(self.path, sp, s_pred, pred, h_pred, v, ld)
    # the pursuit's own curvature from the path itself (its chord), taken out as the feedforward stands for it; the
    # heading interpolated, as a per-point one steps several degrees in a tight bend and makes the steering chatter
    ideal = np.array([np.interp(s_pred, sp, self.path[:, 0]), np.interp(s_pred, sp, self.path[:, 1])])
    k_pp0 = pursuit(self.path, sp, s_pred, ideal, game_heading(float(np.interp(s_pred, sp, self.theta))), v, ld)
    k = k_ff + (k_pp - k_pp0)
    vv = max(v, 1.0)
    cap = min(KAPPA_MAX, A_LAT_MAX / (vv * vv))
    rate = max(RATE_SHARE * self.style["j_lat"] / (vv * vv), 0.03)
    rate *= 1.0 + RATE_OFF_PATH * min(max((self.dev - 0.3) / 0.7, 0.0), 1.0)  # quicker back onto the path once off it
    want = k
    k = float(np.clip(k, -cap, cap))
    k = float(np.clip(k, self.kappa - rate * dt, self.kappa + rate * dt))
    sat = abs(want - k) > max(0.15 * abs(want), 0.002) and v > 2.0
    start = self.timers.get("sat")
    self.timers["sat"] = (start if start is not None else t) if sat else None
    if self.timers["sat"] is not None and t - self.timers["sat"] >= SATURATED_S:
      self._anomaly("tracking_saturated", t, pos, self.s, want=round(want, 4), got=round(k, 4), v=round(v, 1))
    self.kappa = k
    return k

  def _labels(self, route, state: dict, v: float, t: float):
    """The desire the car signals: a lane change from signal_lead s before it until most of the way across, a turn or
    signalled keep as the AI expert signals them."""
    s = self.s
    rs = float(np.interp(s, self.s_path, self.route_s))
    self.target_lane = lane_at(self.keys, rs)
    sec = route.section(segment_of(route, rs))
    self.target_right = round(sec.offset(self.target_lane), 2) if sec is not None and sec.lanes else math.nan
    self.lc_tau = self.lc_dir = math.nan
    side = label = None
    for s0, s1, d, _ in self.changes:
      if s0 - self.style["signal_lead"] * max(v, 3.0) <= s < s0 + LC_DONE * (s1 - s0):
        side, label = d, "laneChange" + d.capitalize()
      if s0 <= s <= s1:
        self.lc_tau = (s - s0) / max(s1 - s0, EPS)
        self.lc_dir = -1.0 if d == "left" else 1.0
    at, heading = route.at, float(state.get("heading") or 0.0)
    for i, m in enumerate(self.mans):
      if i in self.done_mans:
        continue
      d = m.along - at
      over = (d < -TURN_PAST or (d < -TURN_DONE_AFTER and abs(wrap(heading - m.exit_heading)) < TURN_DONE_HEADING)) if m.turn else d < -KEEP_PAST
      if over:
        self.done_mans.add(i)
        continue
      if d <= max(SIGNAL_DIST, v * SIGNAL_TIME):
        side = ("left" if m.desire.endswith("Left") else "right") if m.signal else side
        label = m.desire
      break
    self.indicator, self.label = side, label or "none"

  def _watch(self, route, state: dict, v: float, t: float):
    """Live map anomalies: the car's side over the kerb, the map's lane matching disagreeing with the plan."""
    pos = state["pos"]
    for c in self.corners:
      if abs(route.at - c["along"]) < CORNER_SPAN:
        depth = self._corner_hit(c, pos[:2], math.radians(float(state.get("heading") or 0.0) + 90.0), self.body)
        if depth is not None and depth > KERB_MARGIN:
          self._anomaly("kerb_contact", t, pos, self.s, planned=False, corner=c["side"], depth=round(depth, 2),
                        corner_at=[round(float(v), 1) for v in c["corner"]])
    near_j = self._unclear(route.at)
    if route.off < 30.0 and not route.elsewhere and not near_j and self.s >= self.join_m:
      sec = route.section(route.seg)
      hw = self.half_width
      if sec is not None and sec.lanes and (route.right - hw < sec.edges[0] - KERB_MARGIN or route.right + hw > sec.edges[1] + KERB_MARGIN):
        self._anomaly("kerb_contact", t, pos, self.s, planned=False, right=round(route.right, 2), edges=[round(e, 2) for e in sec.edges])
    m = state.get("laneMap") or {}
    changing = not math.isnan(self.lc_tau)
    sec = route.section(route.seg)
    bad = (not near_j and not changing and m.get("kind") == "own" and m.get("lane") is not None and sec is not None and
           m.get("lanes") == sec.lanes and abs(self.target_lane - round(self.target_lane)) < 0.1 and m["lane"] != round(self.target_lane))
    start = self.timers.get("lane")
    self.timers["lane"] = (start if start is not None else t) if bad else None
    if self.timers["lane"] is not None and t - self.timers["lane"] >= LANE_DISAGREE_S:
      self._anomaly("lane_disagree", t, pos, self.s, plan=round(self.target_lane, 2), matched=[m.get("lane"), m.get("lanes")])


def eased(need: np.ndarray, n: int) -> np.ndarray:
  """A push (right positive) each way reaching at least `need` wherever it's asked, eased in and out over n points
  either side (the largest within n/2 either way, then averaged over n)."""
  out = np.zeros(len(need))
  for sign in (1.0, -1.0):
    x = np.maximum(sign * need, 0.0)
    if not x.any():
      continue
    h = max(n // 2, 1)
    env = x.copy()
    for j in range(1, h + 1):
      env[j:] = np.maximum(env[j:], x[:-j])
      env[:-j] = np.maximum(env[:-j], x[j:])
    k = np.hanning(2 * h + 3)[1:-1]
    out += sign * np.convolve(np.pad(env, h, mode="edge"), k / k.sum(), mode="valid")[:len(need)]
  return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
  e = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
  return list(zip(e[::2], e[1::2], strict=True))


def route_point(route, s: float) -> np.ndarray:
  return np.array([np.interp(s, route.along, route.points[:, 0]), np.interp(s, route.along, route.points[:, 1])])
