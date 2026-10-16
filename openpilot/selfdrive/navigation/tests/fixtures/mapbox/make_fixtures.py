#!/usr/bin/env python3
"""Makes the Mapbox Directions fixtures here: invented routes near 0°N 0°E in the schema and with the lane features of
real UK responses (README.md). No real response is kept; `python make_fixtures.py` rewrites every file.

Lanes are written left to right as in Mapbox: S straight, L left, R right, l slight left, r slight right, U uturn,
N none; `/` joins a lane's arrows, `*` marks it valid (and active) for the route's move, `~b` / `~c` a bus or cycle
lane. A valid lane's valid_indication is its first arrow.
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).parent
METERS_PER_DEGREE = 111319.49
ARROWS = {"S": "straight", "L": "left", "R": "right", "l": "slight left", "r": "slight right", "U": "uturn", "N": "none"}
TURN = {"straight": 0, "slight right": 35, "right": 90, "sharp right": 135, "slight left": -35, "left": -90,
        "sharp left": -135, "uturn": 180}
KMH = {20: 32, 30: 48, 40: 64, 50: 80, 60: 97, 70: 113}  # UK limits as Mapbox gives them
PARAMS = {"steps": "true", "banner_instructions": "true", "annotations": "maxspeed,congestion,distance,duration",
          "overview": "full", "geometries": "polyline6", "alternatives": "false", "language": "en"}
ORDINALS = ["first", "second", "third", "fourth", "fifth"]


def lanes(text: str | None) -> list[dict] | None:
  if text is None:
    return None
  out = []
  for tok in text.split():
    tok, _, access = tok.partition("~")
    valid = tok.endswith("*")
    ind = [ARROWS[a] for a in tok.rstrip("*").split("/")]
    lane = {"active": valid, "valid": valid, **({"valid_indication": ind[0]} if valid else {}), "indications": ind}
    if access:
      lane["access"] = {"designated": [{"b": "bus", "c": "bicycle"}[access]]}
    out.append(lane)
  return out


def encode(points: list[tuple[float, float]]) -> str:
  """(lon, lat) points as a polyline6 (lat, lon order on the wire)."""
  out, prev = [], (0, 0)
  for lon, lat in points:
    cur = (round(lat * 1e6), round(lon * 1e6))
    for v in (cur[0] - prev[0], cur[1] - prev[1]):
      v = ~(v << 1) if v < 0 else v << 1
      while v >= 0x20:
        out.append(chr((0x20 | (v & 0x1f)) + 63))
        v >>= 5
      out.append(chr(v + 63))
    prev = cur
  return "".join(out)


def step(type_, name="", length=0.0, mod=None, mph=30, cls="street", at=None, ints=(), **kw):
  """A step: its maneuver, the road after it (length m), the lanes at the maneuver's own intersection (`at`, with
  `at_flags`), and its other intersections [(m from the step's start, lanes, flags)]."""
  return {"type": type_, "name": name, "length": length, "mod": mod, "mph": mph, "cls": cls, "at": at, "ints": list(ints), **kw}


def instruction(s: dict) -> str:
  t, mod, name = s["type"], s["mod"] or "", s["name"] or s.get("ref", "")
  side = "left" if "left" in mod else "right"
  if t == "depart":
    return f"Drive along {name}."
  if t in ("roundabout", "rotary"):
    return f"Enter the roundabout and take the {ORDINALS[s['exit'] - 1]} exit onto {name}."
  if t == "fork":
    return f"Keep {side} at the fork."
  if t == "off ramp":
    return f"Take the exit on the {side}."
  if t == "on ramp":
    return "Take the ramp."
  if t == "merge":
    return f"Merge {side} onto {name}."
  if t == "arrive":
    return "You have arrived at your destination."
  return f"Turn {mod} onto {name}."


def lonlat(origin: tuple[float, float], px: float, py: float) -> tuple[float, float]:
  return (round(origin[0] + px / METERS_PER_DEGREE, 6), round(origin[1] + py / METERS_PER_DEGREE, 6))


def intersection(s: dict, at: tuple[float, float], gi: int, h: float, h_before: float, lane_text, flags: dict,
                 first: bool) -> dict:
  """An intersection of step s at (lon, lat) `at`, route geometry index gi; the road runs on at bearing h."""
  back, out_b = round((h_before if first else h) + 180) % 360, round(h) % 360
  if s["type"] == "depart":
    bearings, entry, io = [out_b], [True], {"out": 0}
  elif s["type"] == "arrive":
    bearings, entry, io = [back], [True], {"in": 0}
  else:
    bearings = sorted({back, out_b, round(h + 90) % 360})
    entry = [b != back for b in bearings]
    io = {"in": bearings.index(back), "out": bearings.index(out_b)}
  it = {"location": list(at), "geometry_index": gi, "admin_index": 0,
        "is_urban": s["cls"] not in ("motorway", "trunk") and not s.get("rural")}
  if s["type"] not in ("depart", "arrive"):
    o = gi % 7
    it.update(turn_weight=0.5 if first else 0.0, duration=round(2.0 + o, 3), weight=round(2.5 + o, 3))
  it.update(io, entry=entry, bearings=bearings)
  if s["type"] != "arrive":
    it["mapbox_streets_v8"] = {"class": s["cls"]}
  if s["cls"] in ("motorway", "motorway_link"):
    it["classes"] = ["motorway"]
  it.update(flags)
  ln = lanes(lane_text)
  if ln:
    it["lanes"] = ln
  return it


def build(name: str, features: list[str], origin: tuple[float, float], heading: float, steps: list[dict]) -> dict:
  steps = steps + [step("arrive")]
  x, y, h = 0.0, 0.0, heading
  route_pts: list[tuple[float, float]] = []
  seg_d, seg_t, seg_v, seg_c = [], [], [], []
  out_steps = []

  for s in steps:
    h_before = h
    if s["type"] != "depart":
      h = (h + TURN.get(s["mod"] or "straight", 0)) % 360
    offsets = sorted({0.0, s["length"], *[o for o, *_ in s["ints"]], *[float(d) for d in range(20, int(s["length"]), 20)]})
    pts = [lonlat(origin, x + o * math.sin(math.radians(h)), y + o * math.cos(math.radians(h))) for o in offsets]
    first_gi = len(route_pts) - 1 if route_pts else 0
    gi_of = {o: first_gi + i for i, o in enumerate(offsets)}
    route_pts += pts[1:] if route_pts else pts
    v = s["mph"] * 0.44704 * 0.85 if s["mph"] else 13.0
    for a, b in zip(offsets, offsets[1:], strict=False):
      d = b - a
      seg_d.append(round(d, 1))
      seg_t.append(round(d / v, 3))
      seg_v.append({"unit": "km/h", "speed": KMH[s["mph"]]} if s["mph"] and not s.get("unknown_speed") else {"unknown": True})
      seg_c.append("unknown" if s["cls"] in ("street", "service") else "low")

    ints = [intersection(s, pts[0], gi_of[0.0], h, h_before, s["at"], s.get("at_flags", {}), first=True)]
    ints += [intersection(s, pts[offsets.index(o)], gi_of[o], h, h_before, lt, fl, first=False) for o, lt, fl in s["ints"]]
    dur = round(sum(seg_t[len(seg_t) - len(offsets) + 1:]) if len(offsets) > 1 else 0.0, 3)
    man = {"location": ints[0]["location"], "bearing_before": 0 if s["type"] == "depart" else round(h_before) % 360,
           "bearing_after": 0 if s["type"] == "arrive" else round(h) % 360, "instruction": instruction(s), "type": s["type"]}
    if s["mod"]:
      man["modifier"] = s["mod"]
    if "exit" in s:
      man["exit"] = s["exit"]
    st = {"geometry": encode(pts if len(pts) > 1 else pts * 2),
          "mode": "driving", "driving_side": "left", "distance": round(s["length"], 3), "duration": dur,
          "weight": round(dur * 1.1, 3), "duration_typical": round(dur * 1.05, 3), "weight_typical": round(dur * 1.15, 3),
          "name": s["name"], "speedLimitSign": "vienna", "speedLimitUnit": "mph", "maneuver": man, "intersections": ints,
          "bannerInstructions": []}
    for key in ("ref", "destinations", "rotary_name"):
      if key in s:
        st[key] = s[key]
    if "exit" in s:
      st["exits"] = str(s["exit"])
    out_steps.append(st)
    x, y = x + s["length"] * math.sin(math.radians(h)), y + s["length"] * math.cos(math.radians(h))

  for k, st in enumerate(out_steps[:-1]):
    nxt, spec = out_steps[k + 1], steps[k + 1]
    primary = {"text": spec["name"] or spec.get("ref", "") or ("You have arrived at your destination" if spec["type"] == "arrive" else ""),
               "type": spec["type"], "modifier": spec["mod"] or "straight"}
    if spec["type"] in ("roundabout", "rotary"):
      primary.update(degrees=180 + TURN.get(spec["mod"] or "straight", 0), driving_side="left")
    primary["components"] = [{"text": primary["text"], "type": "text"}]
    ln = nxt["intersections"][0].get("lanes")
    d = st["distance"]
    if not ln:
      st["bannerInstructions"] = [{"distanceAlongGeometry": d, "primary": primary}]
      continue
    sub = {"text": "", "components": [{"text": "", "type": "lane", "active": lane["active"], "directions": lane["indications"],
                                       "active_direction": lane.get("valid_indication", lane["indications"][0])} for lane in ln]}
    reach = (3218.688 if d > 3218.688 else 1609.344) if steps[k]["cls"] == "motorway" else 402.336
    if d > reach:
      st["bannerInstructions"] = [{"distanceAlongGeometry": d, "primary": primary},
                                  {"distanceAlongGeometry": reach, "primary": primary, "sub": sub}]
    else:
      st["bannerInstructions"] = [{"distanceAlongGeometry": d, "primary": primary, "sub": sub}]

  dist = round(sum(seg_d), 3)
  dur = round(sum(seg_t), 3)
  named = sorted((s for s in steps if s["name"]), key=lambda s: -s["length"])
  leg = {"via_waypoints": [], "admins": [{"iso_3166_1_alpha3": "GBR", "iso_3166_1": "GB"}],
         "annotation": {"maxspeed": seg_v, "congestion": seg_c, "distance": seg_d, "duration": seg_t},
         "weight": round(dur * 1.1, 3), "duration": dur, "duration_typical": round(dur * 1.05, 3),
         "weight_typical": round(dur * 1.15, 3), "steps": out_steps, "distance": dist,
         "summary": ", ".join(s["name"] for s in named[:2])}
  route = {"weight_name": "auto", "weight": leg["weight"], "duration": dur, "duration_typical": leg["duration_typical"],
           "weight_typical": leg["weight_typical"], "distance": dist, "legs": [leg], "geometry": encode(route_pts)}
  tz = {"offset": "+01:00", "identifier": "Europe/London", "abbreviation": "BST"}
  ends = [route_pts[0], route_pts[-1]]
  names = (steps[0]["name"], steps[-2]["name"])
  waypoints = [{"location": list(p), "name": n, "distance": 3.5, "time_zone": tz} for p, n in zip(ends, names, strict=True)]
  coords = ";".join(f"{lon:.6f},{lat:.6f}" for lon, lat in ends)
  return {"name": name, "features": features, "made": "invented: make_fixtures.py",
          "request": {"path": f"/directions/v5/mapbox/driving-traffic/{coords}", "params": PARAMS},
          "response": {"routes": [route], "waypoints": waypoints, "code": "Ok", "uuid": f"made-{name}"}}


SIG = {"traffic_signal": True}
FIXTURES = [
  ("turn_bay_right", ["right-turn bay: 2 lanes (`S* R`) 250 m before Marsh Lane, 3 (`S R R*`) in the last 33 m",
                      "the turn's own intersection listing 2 (`S R*`)", "lanes valid for each way's own junction"], 0, [
    step("depart", "Orchard Way", 300, ints=[(120, None, {})]),
    step("turn", "Hollin Road", 600, "right", cls="primary", ints=[
      (150, None, SIG), (350, "S* R", {}), (420, "S* R", SIG), (567, "S R R*", {}), (585, "S R R*", {})]),
    step("turn", "Marsh Lane", 400, "right", cls="secondary", at="S R*", at_flags=SIG, ints=[(200, None, {})]),
  ]),
  ("motorway_exit_lane_drop", ["exit with a lane drop: 3 lanes (`S* S S`) 43-270 m before the diverge, 4 at the gore (`l* S S S`)",
                               "lane gain 2 -> 4 after a junction", "banner lanes from 2 miles out"], 90, [
    step("depart", "Kestrel Road", 200, cls="primary", mph=40),
    step("on ramp", "", 300, "slight left", mph=50, cls="motorway_link", ref="M90"),
    step("merge", "", 4000, "slight right", mph=70, cls="motorway", ref="M90", ints=[
      (500, "S* S*", {}), (1000, "S* S*", {}), (1500, "S* S* S* S*", {}), (2500, "S* S* S*", {}), (3730, "S* S S", {}),
      (3957, "S* S S", {})]),
    step("off ramp", "", 600, "slight left", mph=50, cls="motorway_link", at="l* S S S", at_flags={"classes": ["motorway"]},
         destinations="M90 (J3): Fenwick"),
    step("end of road", "Fenwick Road", 500, "left", mph=40, cls="primary", at="L* R"),
  ]),
  ("motorway_exit_shared_lane", ["exit with a shared lane (`l* l/S* S S`)", "a rotary onto the slip road",
                                 "a roundabout at the end of the exit"], 180, [
    step("depart", "Wharf Road", 300, cls="secondary"),
    step("rotary", "", 400, "slight left", cls="motorway_link", mph=50, exit=2, ref="M90", rotary_name="Ash Interchange"),
    step("merge", "", 2600, "slight left", mph=70, cls="motorway", ref="M90", ints=[
      (700, "S* S* S*", {}), (1800, "S* S* S*", {}), (2400, "l* l/S* S", {})]),
    step("off ramp", "", 500, "slight left", mph=50, cls="motorway_link", at="l* l/S* S S", at_flags={"classes": ["motorway"]}),
    step("roundabout", "Denby Lane", 600, "left", cls="secondary", exit=1),
  ]),
  ("two_left_turns_close", ["two left turns 80 m apart, lanes (`L* S S`) for the first"], 0, [
    step("depart", "Linden Street", 400, cls="secondary", ints=[(250, "L* S S", {}), (380, "L* S S", SIG)]),
    step("turn", "Quarry Road", 80, "left", at="L* S S", at_flags=SIG),
    step("turn", "Tanner Row", 300, "left", at_flags={"stop_sign": True}),
  ]),
  ("close_pair_then_fork", ["two maneuvers 10 m apart (left onto Cobb Street, right onto Ferry Road)",
                            "a fork 259 m on (`l* S S`)", "lane gains and drops along the trunk road"], 270, [
    step("depart", "Wren Street", 300),
    step("turn", "Cobb Street", 10, "left", at="L* S"),
    step("turn", "Ferry Road", 259, "right", cls="trunk", mph=40, ints=[(150, "l* S S", {})]),
    step("fork", "Harbour Way", 2500, "slight left", cls="trunk", mph=50, ref="A9", at="l* S S", ints=[
      (800, "S* S*", {}), (1600, "S* S* S*", {}), (2200, "S* S*", {}), (2450, "L* S", {})]),
    step("turn", "Dock Road", 400, "left", at="L* S"),
  ]),
  ("roundabout_entry_gains_lane", ["a roundabout entry gaining a lane at the give way (`S* S*` -> `S* S* S`)",
                                   "a B-road with no lanes"], 45, [
    step("depart", "Ridge Road", 600, cls="primary", mph=40, ints=[(400, "S* S*", {}), (560, "S* S*", {})]),
    step("roundabout", "Abbey Road", 800, "straight", cls="primary", mph=40, exit=2, at="S* S* S", at_flags={"yield_sign": True}),
    step("turn", "Bell Lane", 500, "right", cls="tertiary"),
  ]),
  ("rural_no_lanes", ["no lane data anywhere on 6 km of rural road", "40-60 mph limits, some unknown"], 135, [
    step("depart", "Moor Road", 2500, cls="secondary", mph=60, rural=True, ints=[(900, None, {}), (1700, None, {})]),
    step("turn", "Heath Road", 2000, "left", cls="secondary", mph=40, rural=True, unknown_speed=True, ints=[(1000, None, {})]),
    step("turn", "Fell Lane", 1500, "right", cls="tertiary", mph=50, rural=True, ints=[(600, None, {"railway_crossing": True})]),
  ]),
  ("right_lanes_cycle", ["2 right lanes (`L R R*`) with a cycle lane on the left (`access`)",
                         "a left lane added 21 m before Mill Street (`S* S*` -> `L* S S`)"], 0, [
    step("depart", "Weaver Street", 300, cls="primary", ints=[(150, "S~c L R R*", {}), (280, "S~c L R R*", SIG)]),
    step("turn", "Kings Road", 400, "right", cls="primary", at="S~c L R R*", at_flags=SIG, ints=[(200, "S* S*", {}), (379, "L* S S", {})]),
    step("turn", "Mill Street", 300, "left", at="L* S S"),
  ]),
  ("lanes_vary_bus", ["2 to 5 lanes along one step, added and dropped between junctions", "two right lanes (`R* R*`)",
                      "bus lanes (`access`)"], 90, [
    step("depart", "Canal Road", 900, cls="primary", ints=[
      (100, "S~b S* S* S*", {}), (250, "S* S* S*", {}), (400, "S~b S* S* S*", SIG), (550, "S* S*", {}), (880, "S~b S S R* R*", {})]),
    step("turn", "Priory Street", 400, "right", cls="secondary", at="S~b S S R* R*", at_flags=SIG),
  ]),
  ("lanes_added_midblock", ["a right lane at some junctions of one road only (added mid-block)",
                            "two maneuvers 11 m apart", "a roundabout"], 0, [
    step("depart", "Quay Street", 700, cls="primary", ints=[(200, "S* S*", {}), (350, "S* S* R", {}), (500, "S* S*", {}), (650, "S* S* R", SIG)]),
    step("turn", "Pier Road", 11, "right", at="S S R*", at_flags=SIG),
    step("turn", "Water Lane", 300, "left"),
    step("roundabout", "Marina Way", 400, "left", exit=3),
  ]),
  ("lanes_always_there", ["3-4 lanes for 260 m into a slight right (lanes that have always been there)", "a rotary"], 0, [
    step("depart", "Civic Way", 300),
    step("turn", "Market Street", 400, "left", cls="primary", ints=[(140, "L S* S*", {}), (250, "L S S r*", {}), (330, "L S S r*", SIG)]),
    step("turn", "Elm Road", 500, "slight right", cls="primary", at="L S S r*", at_flags=SIG),
    step("rotary", "Grove Road", 400, "slight left", exit=1, rotary_name="Grove Circus"),
  ]),
  ("lanes_last_5m", ["the junction's 4 lanes only in the last 5 m (`L R*` -> `L L R* R`)"], 0, [
    step("depart", "Ash Road", 500, cls="primary", ints=[(200, "L R*", {}), (400, "L R*", {}), (495, "L L R* R", {})]),
    step("turn", "Park Street", 300, "right", cls="primary", at="L L R* R", at_flags=SIG),
  ]),
]


def main():
  for p in HERE.glob("*.json"):
    p.unlink()
  for i, (name, features, heading, steps) in enumerate(FIXTURES):
    rec = build(name, features, (0.02 * i, 0.0), heading, steps)
    (HERE / f"{name}.json").write_text(json.dumps(rec, separators=(",", ":")) + "\n")


if __name__ == "__main__":
  main()
