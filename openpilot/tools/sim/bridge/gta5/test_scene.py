import random

from openpilot.tools.sim.bridge.gta5 import gta5_scene


def test_interior_only_for_the_model_3():
  for seed in range(300):
    choice = gta5_scene.pick(seed)
    v = choice["vehicle"]
    assert ("interior" in v) == ("chrome" in v) == (v["car"] == "model3")
    want = [{"type": "trim", "interior": v["interior"], "chrome": v["chrome"]}] if "interior" in v else []
    assert gta5_scene.trim_commands(choice) == want
    assert all(c["type"] != "trim" for c in gta5_scene.commands(choice))  # sent after the swap instead


def test_interior_picks_both():
  rng = random.Random(1)
  picks = {gta5_scene.pick_interior(rng) for _ in range(100)}
  assert picks == {gta5_scene.INTERIORS["white"][1], gta5_scene.INTERIORS["black"][1]}


def test_trim_picks_both():
  rng = random.Random(1)
  assert {gta5_scene.pick_chrome(rng) for _ in range(100)} == {True, False}


def test_interior_index():
  assert gta5_scene.interior_index("white") == 134
  assert gta5_scene.interior_index("black") == 12
  assert gta5_scene.interior_index(2.0) == 2
