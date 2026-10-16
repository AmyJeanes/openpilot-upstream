"""The Mapbox response fixtures (fixtures/mapbox: invented routes in real responses' schema, README.md there): what the
lane guidance and the GTA emulator rely on."""
import json
import os
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "mapbox"
NAMES = sorted(p.stem for p in FIXTURES.glob("*.json"))
INDICATIONS = {"left", "slight left", "sharp left", "straight", "slight right", "right", "sharp right", "uturn", "none"}


def load(name: str) -> dict:
  return json.loads((FIXTURES / f"{name}.json").read_text())


def test_fixtures_present():
  assert len(NAMES) >= 10


@pytest.mark.parametrize("name", NAMES)
def test_no_token(name):
  text = (FIXTURES / f"{name}.json").read_text()
  assert "access_token" not in load(name)["request"]["params"]
  assert "access_token=" not in text
  token = os.environ.get("MAPBOX_TOKEN")
  assert not token or token not in text


@pytest.mark.parametrize("name", NAMES)
def test_lanes(name):
  """Lanes only on intersections, valid_indication only on valid lanes; banner lanes are the maneuver's own."""
  route = load(name)["response"]["routes"][0]
  steps = route["legs"][0]["steps"]
  for k, step in enumerate(steps):
    for it in step["intersections"]:
      for lane in it.get("lanes", []):
        assert set(lane["indications"]) <= INDICATIONS
        assert ("valid_indication" in lane) == lane["valid"]
        assert not lane["active"] or lane["valid"]
    for banner in step["bannerInstructions"]:
      comps = [c for c in (banner.get("sub") or {}).get("components", []) if c["type"] == "lane"]
      if comps:
        lanes = steps[k + 1]["intersections"][0]["lanes"]
        assert [c["directions"] for c in comps] == [lane["indications"] for lane in lanes]
        assert [c["active"] for c in comps] == [lane["active"] for lane in lanes]
        assert banner["distanceAlongGeometry"] <= step["distance"] + 1.0
  assert "lanes" not in json.dumps([s["maneuver"] for s in steps])


@pytest.mark.parametrize("name", NAMES)
def test_annotations(name):
  leg = load(name)["response"]["routes"][0]["legs"][0]
  ann = leg["annotation"]
  assert len(ann["maxspeed"]) == len(ann["congestion"]) == len(ann["distance"])
  for m in ann["maxspeed"]:
    assert m in ({"unknown": True}, {"none": True}) or (set(m) == {"speed", "unit"} and m["unit"] in ("km/h", "mph"))


def test_bay_is_an_extra_lane():
  """Into Marsh Lane: two lanes 250 m out, three (a second right lane) just before the junction."""
  steps = load("turn_bay_right")["response"]["routes"][0]["legs"][0]["steps"]
  k = next(i for i, s in enumerate(steps) if s["name"] == "Marsh Lane")
  approach = [[x["indications"] for x in it["lanes"]] for it in steps[k - 1]["intersections"] if it.get("lanes")]
  assert approach[0] == [["straight"], ["right"]]
  assert approach[-1] == [["straight"], ["right"], ["right"]]


def test_no_lanes_on_a_rural_road():
  route = load("rural_no_lanes")["response"]["routes"][0]
  assert not any(it.get("lanes") for s in route["legs"][0]["steps"] for it in s["intersections"])
