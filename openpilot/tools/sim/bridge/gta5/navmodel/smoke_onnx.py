#!/usr/bin/env python3
"""A CPU smoke test of a nav model's ONNX (comma's big_driving_supercombo.onnx, nav-model-integration a03bb1d, the rl
nav model): the inputs the graph declares, then one run with every input zero but a GTA map frame and the navigation
vector for a right turn ahead, and again with those two zero, to see that the nav inputs reach the outputs.

  python smoke_onnx.py big_driving_supercombo.onnx [--tiles ~/gta5map/gta5_roads.pmtiles]

Needs onnxruntime (PyPI) and a few GB of RAM. Without --tiles the map is a made-up road and route.
"""
import argparse
import json
from pathlib import Path

import numpy as np

DTYPES = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(uint8)": np.uint8, "tensor(int64)": np.int64}


def gta_inputs(tiles: str | None) -> tuple[np.ndarray, np.ndarray]:
  """A map frame and the navigation vector 150 m along the city fixture route."""
  from openpilot.tools.sim.bridge.gta5.map.router import Route
  from openpilot.tools.sim.bridge.gta5.navmodel.nav_input import ModelNav, navigation

  r = json.loads((Path(__file__).parent / "fixtures" / "routes.json").read_text())["city"]
  route = Route(np.array(r["points"], dtype=float), r["maneuvers"])
  nav = ModelNav(route)
  s = 150.0
  p = np.array([np.interp(s, route.along, route.points[:, 0]), np.interp(s, route.along, route.points[:, 1])])
  for k in np.arange(0.0, s, 10.0):
    nav.all_maneuvers_at([np.interp(k, route.along, route.points[:, 0]), np.interp(k, route.along, route.points[:, 1])])
  vector = navigation(nav.all_maneuvers_at(p))
  if tiles:
    from openpilot.selfdrive.navigation.local_tiles import LocalTiles
    from openpilot.selfdrive.navigation.pmtiles import PMTiles
    from openpilot.selfdrive.navigation.renderer import MapRenderer
    from openpilot.tools.sim.bridge.gta5.navmodel.render_frames import frames_along
    image = frames_along(MapRenderer(PMTiles(LocalTiles(tiles))), route.points, [s])[0][1]
  else:
    image = np.zeros((256, 256), dtype=np.uint8)
    image[:, 120:137] = 255
    image[60:77, 128:] = 255
    image[60:192, 126:131] = 128
  return image, vector


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("onnx")
  ap.add_argument("--tiles")
  args = ap.parse_args()
  import onnxruntime as ort

  sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
  print("inputs:")
  for i in sess.get_inputs():
    print(f"  {i.name:24s} {i.type:16s} {i.shape}")
  print("outputs:", [(o.name, o.shape) for o in sess.get_outputs()])
  image, vector = gta_inputs(args.tiles)
  print("navigation: type", int(np.argmax(vector[:46])) if vector[:46].any() else None, "distance/100", float(vector[46]))

  def feeds(nav: bool) -> dict:
    out = {}
    for i in sess.get_inputs():
      shape = [d if isinstance(d, int) else 1 for d in i.shape]
      x = np.zeros(shape, dtype=DTYPES.get(i.type, np.float32))
      if nav and i.name == "nav_map":
        x[...] = image.reshape([1] * (len(shape) - 2) + [256, 256]) if x.size == image.size else 0
      elif nav and i.name == "navigation":
        x[...] = vector.reshape(shape) if x.size == vector.size else 0
      out[i.name] = x
    return out

  names = [i.name for i in sess.get_inputs()]
  if "nav_map" not in names or "navigation" not in names:
    print("WARNING: no nav_map / navigation input:", names)
  with_nav = sess.run(None, feeds(True))
  without = sess.run(None, feeds(False))
  for o, a, b in zip(sess.get_outputs(), with_nav, without, strict=True):
    print(f"  {o.name}: finite={bool(np.isfinite(a).all())} max|with - without| = {float(np.abs(a.astype(np.float64) - b).max()):.4g}")


if __name__ == "__main__":
  main()
