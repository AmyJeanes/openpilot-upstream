"""Scene choices for varied training data: the weather, time of day, traffic, the car and its colours, and the camera mount,
picked per drive (`gta5_cmd.py randomise`) or set one at a time (`gta5_cmd.py world|traffic|vehicle|mount`).

Nothing here talks to the game: `pick()` makes a choice and `commands()` the plugin commands for it."""
import random
from dataclasses import dataclass

MODEL_3 = "0x0040B009"  # the add-on Tesla Model 3, which the bridge's car params are for


@dataclass(frozen=True)
class Car:
  name: str  # the game's model name
  hash: str  # as the state's vehicle.model shows it
  kind: str
  weight: float  # its share of the cars picked other than the Model 3
  colours: list  # the game's own colour sets for it (carcols): (primary, secondary, pearlescent)


# common, realistic road cars of the base game (DurtyFree's gta-v-data-dumps vehicles.json), all with windscreen and
# driver's seat bones for the dashcam mount
CARS = [
  Car("premier", "0x8FB66F9B", "sedan", 1, [
      (0, 0, 5), (4, 0, 111), (5, 0, 111), (6, 0, 111), (7, 0, 67), (10, 0, 3), (10, 0, 6), (28, 0, 28), (35, 0, 36), (54, 0, 54), (69, 0, 74),
      (97, 0, 97), (111, 0, 0)]),
  Car("asea", "0x94204D89", "sedan", 1, [
      (0, 0, 7), (3, 3, 5), (4, 4, 111), (6, 0, 111), (7, 7, 6), (29, 34, 28), (31, 27, 31), (49, 0, 49), (52, 49, 52), (62, 0, 63), (111, 0, 111),
      (112, 0, 112)]),
  Car("stanier", "0xA7EDE74D", "sedan", 1, [
      (0, 0, 10), (1, 1, 7), (2, 2, 5), (3, 3, 24), (4, 4, 111), (6, 6, 111), (31, 31, 32), (34, 34, 33), (37, 37, 38), (67, 67, 111),
      (69, 69, 67), (93, 93, 107), (111, 111, 0)]),
  Car("fugitive", "0x71CB2FFB", "sedan", 1, [
      (0, 0, 1), (1, 0, 5), (3, 0, 5), (4, 0, 111), (5, 0, 111), (6, 0, 111), (6, 0, 132), (31, 0, 29), (34, 0, 36), (51, 0, 53), (66, 0, 68),
      (98, 0, 99), (111, 0, 111)]),
  Car("tailgater", "0xC3DDFDCE", "sedan", 1, [
      (0, 0, 10), (1, 1, 10), (2, 2, 7), (3, 3, 5), (4, 4, 111), (5, 5, 111), (6, 6, 111), (27, 27, 36), (32, 32, 28), (61, 61, 67), (64, 64, 54),
      (111, 111, 111)]),
  Car("schafter2", "0xB52B5113", "sedan", 1, [
      (0, 0, 10), (1, 1, 7), (2, 2, 5), (3, 3, 24), (4, 4, 111), (6, 6, 111), (31, 31, 41), (34, 34, 33), (37, 37, 38), (67, 67, 111),
      (69, 69, 67), (93, 93, 107), (111, 111, 0)]),
  Car("washington", "0x69F06B57", "sedan", 1, [
      (0, 0, 18), (1, 0, 77), (1, 1, 5), (2, 2, 5), (4, 0, 111), (5, 23, 111), (6, 6, 111), (7, 18, 5), (27, 47, 102), (29, 29, 36), (33, 38, 36),
      (49, 49, 65), (64, 86, 74), (66, 66, 80), (67, 68, 111), (105, 94, 106), (111, 0, 0), (111, 45, 0)]),
  Car("intruder", "0x34DD8AA1", "sedan", 1, [
      (0, 0, 2), (2, 0, 4), (4, 0, 111), (5, 0, 111), (29, 0, 29), (32, 0, 29), (50, 0, 52), (66, 0, 68), (95, 0, 98), (111, 0, 111)]),
  Car("asterope", "0x8E9254FB", "sedan", 1, [
      (0, 0, 1), (2, 68, 5), (3, 3, 5), (4, 4, 111), (5, 0, 111), (6, 0, 5), (9, 0, 5), (10, 0, 5), (10, 77, 5), (30, 0, 35), (37, 0, 102),
      (52, 0, 52), (62, 0, 63), (62, 0, 65), (67, 0, 111), (74, 0, 74), (95, 0, 95), (111, 0, 111)]),
  Car("surge", "0x8F0E3594", "sedan", 1, [
      (0, 0, 74), (1, 0, 7), (2, 0, 67), (3, 0, 132), (4, 0, 111), (5, 0, 111), (6, 0, 111), (9, 0, 67), (33, 0, 27), (61, 0, 68), (105, 0, 106),
      (111, 0, 0)]),
  Car("dilettante", "0xBC993509", "hatchback", 1, [
      (0, 0, 1), (4, 0, 111), (5, 0, 111), (6, 0, 111), (7, 0, 111), (9, 123, 5), (29, 123, 37), (33, 0, 32), (73, 0, 77), (74, 123, 74),
      (93, 123, 93), (104, 123, 105), (111, 123, 111)]),
  Car("blista", "0xEB70965F", "hatchback", 1, [
      (0, 0, 1), (1, 0, 4), (4, 0, 5), (4, 0, 111), (6, 0, 111), (7, 0, 5), (8, 0, 5), (30, 0, 38), (33, 0, 35), (41, 0, 41), (61, 0, 65),
      (104, 0, 37), (111, 0, 111)]),
  Car("prairie", "0xA988D3A2", "hatchback", 1, [
      (0, 0, 0), (1, 0, 2), (6, 0, 7), (29, 0, 30), (36, 0, 89), (56, 0, 56), (68, 0, 70), (106, 0, 88), (111, 0, 111)]),
  Car("serrano", "0x4FB1A214", "suv", 1, [
      (0, 0, 0), (1, 1, 4), (2, 2, 7), (3, 3, 23), (4, 4, 111), (5, 5, 111), (11, 11, 6), (31, 31, 36), (34, 34, 44), (66, 66, 67), (71, 71, 74),
      (73, 73, 74), (111, 111, 0), (111, 111, 111)]),
  Car("seminole", "0x48CECED3", "suv", 1, [
      (0, 0, 0), (0, 2, 7), (2, 2, 18), (3, 3, 5), (4, 4, 111), (6, 6, 111), (10, 10, 6), (34, 6, 34), (35, 35, 35), (53, 58, 51), (69, 0, 66),
      (97, 1, 101), (111, 3, 0), (111, 111, 0), (113, 113, 106)]),
  Car("radi", "0x9D96B45B", "suv", 1, [
      (0, 0, 18), (0, 0, 64), (1, 1, 5), (2, 0, 5), (4, 4, 111), (5, 5, 111), (10, 0, 5), (30, 30, 36), (31, 31, 28), (34, 0, 44), (34, 34, 35),
      (49, 49, 74), (62, 1, 77), (64, 64, 78), (66, 0, 86), (111, 0, 0)]),
  Car("habanero", "0x34B7390F", "suv", 1, [
      (0, 0, 10), (1, 1, 18), (2, 2, 4), (3, 3, 23), (4, 0, 111), (6, 6, 111), (8, 0, 5), (28, 0, 36), (33, 22, 41), (62, 86, 65), (66, 77, 86),
      (70, 87, 70), (111, 0, 1), (111, 49, 111)]),
  Car("fq2", "0xBC32A33B", "suv", 1, [
      (0, 0, 3), (1, 13, 3), (3, 3, 5), (4, 13, 111), (6, 6, 111), (7, 13, 5), (10, 13, 5), (28, 13, 28), (49, 49, 53), (52, 52, 52), (62, 14, 78),
      (66, 66, 68), (69, 69, 51), (103, 103, 109), (105, 105, 106), (111, 14, 111)]),
  Car("granger", "0x9628879C", "suv", 1, [
      (1, 1, 4), (2, 2, 4), (2, 2, 5), (5, 13, 111), (6, 6, 111), (31, 0, 4), (33, 33, 33), (61, 13, 2), (62, 62, 74), (94, 13, 95),
      (103, 103, 105), (111, 111, 111), (116, 116, 109)]),
  Car("cavalcade2", "0xD0EB2BE5", "suv", 1, [
      (0, 0, 1), (4, 4, 111), (6, 6, 111), (7, 7, 5), (11, 11, 5), (34, 34, 32), (66, 66, 66), (67, 67, 111), (94, 94, 97), (101, 101, 103),
      (111, 111, 111)]),
  Car("landstalker", "0x4BA4E8DC", "suv", 1, [
      (0, 0, 8), (1, 0, 4), (2, 0, 5), (4, 2, 111), (5, 0, 111), (5, 6, 111), (10, 0, 5), (32, 1, 37), (32, 32, 33), (33, 33, 35), (61, 0, 67),
      (73, 7, 74), (111, 111, 0)]),
  Car("minivan", "0xED7EADA4", "van", 1, [
      (0, 0, 4), (3, 3, 4), (4, 4, 111), (5, 4, 111), (5, 5, 111), (6, 1, 111), (7, 0, 4), (10, 0, 4), (34, 34, 31), (63, 66, 65), (101, 0, 104),
      (104, 4, 104), (106, 106, 106), (111, 111, 0)]),
  Car("speedo", "0xCFB3870C", "van", 0.5, [(0, 0, 2), (4, 0, 110), (5, 111, 110), (7, 0, 5), (10, 0, 5), (27, 0, 28), (73, 0, 6), (111, 0, 0)]),
  Car("bobcatxl", "0x3FC5D440", "pickup", 0.5, [
      (0, 0, 10), (1, 1, 4), (2, 2, 4), (5, 5, 111), (6, 6, 5), (31, 0, 5), (34, 1, 5), (49, 51, 51), (52, 107, 52), (72, 68, 68), (77, 77, 77),
      (100, 113, 106), (103, 106, 103)]),
]
CARS_BY_NAME = {c.name: c for c in CARS}
MODEL_3_NAMES = {"model3", "tesla", "m3", MODEL_3.lower()}

# real cars' colours, roughly by their share on the road, as palette indices (DurtyFree's vehicleColors.json)
COLOUR_GROUPS = {
  "white": (25, [111, 112]),
  "black": (20, [0, 1, 2, 11]),
  "grey": (18, [3, 6, 7, 8, 10]),
  "silver": (12, [4, 5, 9]),
  "blue": (9, [61, 62, 63, 64, 65, 66, 68, 69, 73, 74]),
  "red": (8, [27, 28, 29, 30, 31, 34, 35]),
  "brown": (4, [93, 94, 95, 97, 98, 99, 102, 105, 106, 107]),
  "green": (2, [49, 50, 51, 52, 53, 54]),
  "orange": (2, [36, 37, 38, 88, 89, 90]),
}
WHEEL = 156  # the game's default alloy colour

WEATHERS = {"EXTRASUNNY": 25, "CLEAR": 22, "CLOUDS": 18, "OVERCAST": 12, "SMOG": 5, "FOGGY": 5, "CLEARING": 4, "RAIN": 7, "THUNDER": 2}
TRAFFIC_LEVELS = {0.3: 15, 0.6: 25, 1.0: 40, 1.5: 20}  # vehicle density multipliers
DROP = (0.05, 0.10)  # m, the dashcam's range below the roof
JITTER_M, JITTER_DEG = 0.02, 0.5


def weighted(rng: random.Random, table: dict):
  return rng.choices(list(table), weights=list(table.values()))[0]


def model_for(name: str) -> str:
  """The plugin's model= for a car: the Model 3's hash, a curated car's name, or anything the game knows (a name or 0x...)."""
  return MODEL_3 if name.lower() in MODEL_3_NAMES else name


def pick_car(rng: random.Random, model_share: float = 0.6) -> str:
  if rng.random() < model_share:
    return "model3"
  return rng.choices([c.name for c in CARS], weights=[c.weight for c in CARS])[0]


def pick_colours(rng: random.Random, car: str = "") -> dict:
  """Half the time one of the model's own colour sets (when it's a curated car), else a real-world colour: secondary the
  same or black trim, pearlescent the same or a neighbour. Dirt 0-15, mostly clean."""
  known = CARS_BY_NAME.get(car)
  if known is not None and rng.random() < 0.5:
    primary, secondary, pearl = rng.choice(known.colours)
    group = "default"
  else:
    group = weighted(rng, {g: w for g, (w, _) in COLOUR_GROUPS.items()})
    shades = COLOUR_GROUPS[group][1]
    primary = rng.choice(shades)
    secondary = primary if rng.random() < 0.7 else 0
    pearl = primary if rng.random() < 0.5 else rng.choice(shades) if rng.random() < 0.6 else rng.choice([5, 111])
  return {"primary": primary, "secondary": secondary, "pearl": pearl, "wheel": WHEEL, "dirt": round(rng.triangular(0, 8, 1), 1),
          "group": group}


def pick_time(rng: random.Random) -> tuple[int, int]:
  """75% day (07-19), 10% dawn or dusk (05-07, 19-21), 15% night (21-05)."""
  r = rng.random()
  if r < 0.75:
    hour = rng.uniform(7, 19)
  elif r < 0.85:
    hour = rng.choice([rng.uniform(5, 7), rng.uniform(19, 21)])
  else:
    hour = rng.uniform(21, 29) % 24
  return int(hour), int(hour % 1 * 60)


def pick_weather(rng: random.Random) -> tuple[str, float]:
  """Mostly clear or cloudy, some fog, smog and rain; rain at the weather's own level (-1) or a set one."""
  weather = weighted(rng, WEATHERS)
  rain = rng.choice([-1, -1, 0.25, 0.5, 0.8]) if weather in ("RAIN", "THUNDER") else -1
  return weather, rain


def pick_traffic(rng: random.Random, hour: int) -> dict:
  """Vehicle density 0.3-1.5 (mostly normal), pedestrians 0.3-1.2, parked cars 0.5-1; halved for vehicles and people at
  night (22-06)."""
  night = hour >= 22 or hour < 6
  scale = 0.5 if night else 1.0
  return {"vehicles": round(weighted(rng, TRAFFIC_LEVELS) * scale, 2), "peds": round(rng.uniform(0.3, 1.2) * scale, 2),
          "parked": round(rng.uniform(0.5, 1.0), 2)}


def pick_mount(rng: random.Random, mode: str = "dashcam", jitter: bool = True) -> dict:
  """The dashcam drop 5-10 cm below the roof; jitter uniform within +-2 cm and +-0.5 deg."""
  out = {"mode": mode}
  if mode == "dashcam":
    out["drop"] = round(rng.uniform(*DROP), 3)
  j = (lambda r: round(rng.uniform(-r, r), 4)) if jitter else (lambda r: 0.0)
  out.update({"dx": j(JITTER_M), "dy": j(JITTER_M), "dz": j(JITTER_M), "pitch": j(JITTER_DEG), "yaw": j(JITTER_DEG)})
  return out


def pick(seed: int, model_share: float = 0.6, mount: str = "mixed", jitter: bool = True, vehicle: bool = True,
         world: bool = True, traffic: bool = True, comma_share: float = 0.5) -> dict:
  """A drive's scene, the same for the same seed and options; parts turned off are left as they are."""
  rng = random.Random(seed)
  hour, minute = pick_time(rng)
  weather, rain = pick_weather(rng)
  car = pick_car(rng, model_share)
  choice: dict = {"seed": seed}
  if world:
    choice["world"] = {"hour": hour, "minute": minute, "weather": weather, "rain": rain}
  if traffic:
    choice["traffic"] = pick_traffic(rng, hour)
  if vehicle:
    choice["vehicle"] = {"car": car, **pick_colours(rng, car)}
  if mount == "mixed":
    # the comma mount is where openpilot's test car carries its camera; on other cars it can sit in the glass
    mount = "comma" if vehicle and car == "model3" and rng.random() < comma_share else "dashcam"
  if mount in ("dashcam", "comma"):
    choice["mount"] = pick_mount(rng, mount, jitter)
  return choice


def commands(choice: dict) -> list[dict]:
  """The plugin commands for a choice, the car first so its model loads while the rest go."""
  out = []
  if "vehicle" in choice:
    v = choice["vehicle"]
    out.append({"type": "vehicle", "model": model_for(v["car"]), **{k: v[k] for k in ("primary", "secondary", "pearl", "wheel", "dirt") if k in v}})
  if "world" in choice:
    out.append({"type": "world", **choice["world"], "transition": 0, "freeze": 1})
  if "traffic" in choice:
    out.append({"type": "traffic", "on": 1, **choice["traffic"]})
  if "mount" in choice:
    out.append({"type": "mount", **choice["mount"]})
  return out
