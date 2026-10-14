"""Other traffic for the map driver (gta5_mapdrive.py): the vehicles and pedestrians the plugin reports around the car
(its state's nearby, as MapDrive._vehicles puts them in the world) tracked from one report to the next, and each one's
motion predicted a few seconds on.

Tracking: the plugin reports every 0.1 s with no identities (or, from a plugin that sends them, each vehicle's entity
handle, which is used as is), so each report's bodies are matched to the tracks by where those were predicted to be
now, nearest first, within TRACK_GATE m (more for faster ones); one unmatched is a new track, and a track without a
reading for TRACK_FORGET s is dropped. A track keeps when it was first seen, since when it has stood still (below
STILL m/s), whether it has moved since it was seen, its yaw rate and its acceleration along its heading (both smoothed).

Prediction (Traffic.predict): a vehicle running along the road the path runs on (its heading within ALONG_ANGLE of the
path's, or of the path's reversed: oncoming), at no more than ALONG_OFF m beside it, keeps to the road: it goes on along
the path at its offset, at its speed (its braking carried on to a stop), until the path reaches a junction or turn of
our route (a place our way may leave the road it's on), from where it goes straight on. A vehicle turning off the road
(its yaw rate off the path's by more than TURN_OFF), or across it, goes on at its yaw rate (fading over YAW_FADE s) and
speed. A pedestrian walks straight on."""
import math

import numpy as np

TRACK_GATE = 3.0  # m from where a track was predicted to be that a reading may be and still be it
TRACK_GATE_V = 0.3  # s of its speed more
TRACK_FORGET = 1.0  # s without a reading before a track is dropped
STILL = 0.5  # m/s: slower is standing still
MOVED = 1.0  # m/s: a track that has gone faster has moved since it was seen
YAW_TAU = 0.3  # s the yaw rate and acceleration are smoothed over
YAW_MAX = 0.6  # rad/s at most
YAW_FADE = 2.0  # s a turn's yaw rate fades out over, predicting
ACCEL_MIN = -6.0  # m/s^2: a harder reading is noise (a bump, a contact)
ALONG_ANGLE = 30.0  # deg off the path's heading (or its reverse) a vehicle running along its road heads at most
ALONG_OFF = 15.0  # m beside the path at most
TURN_OFF = 0.15  # rad/s between a vehicle's yaw rate and the path's (its curvature x the vehicle's speed): turning off


def wrap(a: float) -> float:
  """An angle in radians into -pi..pi."""
  return (a + math.pi) % (2 * math.pi) - math.pi


def body_heading(c: np.ndarray) -> float:
  """A body's heading (radians, world, x east y north) from its corners (rear left, rear right, front right, front
  left: the order MapDrive._vehicles gives)."""
  fwd = c[3] - c[0]
  return math.atan2(fwd[1], fwd[0])


class Track:
  def __init__(self, tid: int, car: dict, t: float):
    self.id = tid
    self.ped = bool(car.get("ped"))
    self.first = t
    self.t = t  # its reading's time
    self.seen = t  # when a report last had it (one the same as the last is no new reading, but it's still there)
    self.c = car["c"]
    self.vel = car["vel"]
    self.yaw = 0.0
    self.accel = 0.0
    self.speed = float(np.hypot(*self.vel))
    self.heading = self.course()
    self.still_since: float | None = t if self.speed < STILL else None
    self.moved = self.speed >= MOVED
    self.handle = car.get("handle")

  def course(self) -> float:
    """Its heading: its body's (a pedestrian's: where it walks), else where it's going."""
    if self.ped:
      return math.atan2(self.vel[1], self.vel[0]) if self.speed > 0.2 else 0.0
    return body_heading(self.c)

  def centre(self, t: float) -> np.ndarray:
    """Where it should be at t, straight on from its reading."""
    return self.c.mean(axis=0) + self.vel * (t - self.t)

  def update(self, car: dict, t: float):
    dt = t - self.t
    speed = float(np.hypot(*car["vel"]))
    heading = body_heading(car["c"]) if not self.ped else (math.atan2(car["vel"][1], car["vel"][0]) if speed > 0.2 else self.heading)
    if dt > 1e-3:
      k = 1.0 - math.exp(-dt / YAW_TAU)
      yaw = wrap(heading - self.heading) / dt if speed > 1.0 else 0.0
      self.yaw += k * (float(np.clip(yaw, -YAW_MAX, YAW_MAX)) - self.yaw)
      acc = (speed - self.speed) / dt
      if acc > ACCEL_MIN:
        self.accel += k * (min(acc, 4.0) - self.accel)
    self.c, self.vel, self.t, self.seen, self.speed, self.heading = car["c"], car["vel"], t, t, speed, heading
    if speed < STILL:
      self.still_since = self.still_since if self.still_since is not None else t
    else:
      self.still_since = None
    self.moved = self.moved or speed >= MOVED


class Traffic:
  """The tracks of what the plugin reports (update each new report), as MapDrive._vehicles' dicts gain "id", "seen"
  (s since first seen), "still" (s standing still, 0 moving), "moved", "yaw" and "accel"."""
  def __init__(self):
    self.tracks: dict[int, Track] = {}
    self.next_id = 1

  def update(self, cars: list[dict], t: float):
    for tid in [k for k, tr in self.tracks.items() if t - tr.seen > TRACK_FORGET]:
      del self.tracks[tid]
    free = dict(self.tracks)
    by_handle = {tr.handle: tid for tid, tr in free.items() if tr.handle is not None}
    pairs = []
    for n, car in enumerate(cars):
      h = car.get("handle")
      if h is not None and h in by_handle:
        pairs.append((-1.0, n, by_handle[h]))
        continue
      c = car["c"].mean(axis=0)
      for tid, tr in free.items():
        if tr.ped != bool(car.get("ped")) or (h is not None and tr.handle is not None):
          continue
        d = float(np.hypot(*(c - tr.centre(t))))
        if d < TRACK_GATE + TRACK_GATE_V * tr.speed:
          pairs.append((d, n, tid))
    pairs.sort()
    used_cars, used_tracks = set(), set()
    for _, n, tid in pairs:
      if n in used_cars or tid in used_tracks:
        continue
      used_cars.add(n)
      used_tracks.add(tid)
      self.tracks[tid].update(cars[n], t)
      self._mark(cars[n], self.tracks[tid], t)
    for n, car in enumerate(cars):
      if n not in used_cars:
        tr = self.tracks[self.next_id] = Track(self.next_id, car, t)
        self.next_id += 1
        self._mark(car, tr, t)

  def again(self, cars: list[dict], t: float):
    """The same report as last time (the plugin's places unchanged, as when everything stands still): its tracks are
    still there, and the times they've stood grow."""
    for car in cars:
      tr = self.tracks.get(car.get("id"))
      if tr is not None:
        tr.seen = t
        self._mark(car, tr, t)

  @staticmethod
  def _mark(car: dict, tr: Track, t: float):
    car.update(id=tr.id, seen=t - tr.first, still=(t - tr.still_since) if tr.still_since is not None else 0.0, moved=tr.moved,
               yaw=tr.yaw, accel=tr.accel)


def rect(centre: np.ndarray, heading: float, length: float, width: float) -> np.ndarray:
  """A body's corners (rear left, rear right, front right, front left) from its middle, heading and size."""
  f = np.array([math.cos(heading), math.sin(heading)])
  r = np.array([math.sin(heading), -math.cos(heading)])
  hl, hw = length / 2, width / 2
  return np.array([centre - f * hl - r * hw, centre - f * hl + r * hw, centre + f * hl + r * hw, centre + f * hl - r * hw])


def size(c: np.ndarray) -> tuple[float, float]:
  """A body's length and width from its corners."""
  return float(np.hypot(*(c[3] - c[0]))), float(np.hypot(*(c[1] - c[0])))


def ctra(car: dict, t0: float, times: np.ndarray) -> np.ndarray:
  """Its corners at each of `times` (s on from t0): on at its yaw rate (fading over YAW_FADE s) and speed (its braking
  carried on to a stop, no speeding up assumed)."""
  c = car["c"]
  dt0 = t0 - car["t"]
  speed = float(np.hypot(*car["vel"]))
  if car.get("ped") or speed < 0.3:
    return np.stack([c + car["vel"] * (dt0 + dt) for dt in times])
  length, width = size(c)
  h0 = body_heading(c)
  # where its body heads and where it's going differ in a slide: go where it's going
  hv = math.atan2(car["vel"][1], car["vel"][0])
  if abs(wrap(hv - h0)) > math.pi / 2:  # backing up: the body's way, reversed
    return np.stack([c + car["vel"] * (dt0 + dt) for dt in times])
  yaw, acc = float(car.get("yaw") or 0.0), min(float(car.get("accel") or 0.0), 0.0)
  out = []
  p, h, v, tt = c.mean(axis=0), hv, speed, 0.0
  step = 0.05
  targets = list(np.asarray(times) + dt0)
  k = 0
  while k < len(targets):
    if tt >= targets[k] - 1e-9:
      out.append(rect(p, h, length, width))
      k += 1
      continue
    d = min(step, targets[k] - tt)
    w = yaw * math.exp(-tt / YAW_FADE)
    v = max(v + acc * d, 0.0)
    p = p + v * d * np.array([math.cos(h), math.sin(h)])
    h += w * d
    tt += d
  return np.stack(out) if out else np.zeros((0, 4, 2))
