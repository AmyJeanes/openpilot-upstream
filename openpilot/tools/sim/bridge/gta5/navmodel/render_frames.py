#!/usr/bin/env python3
"""The nav model's map frames along a GTA route, rendered by comma's renderer as mapsd renders them (selfdrive/
navigation/mapsd.py: the pose on the route at the car's projected progress, headed along the route), from a local
PMTiles file (build_pmtiles.py):

  python render_frames.py gta5_roads.pmtiles out_dir [--route fixtures/routes.json:city] [--every 250]

The route goes into the frame the tiles were built in (their metadata's lat0).

Writes one PNG per frame (frame_<route>_<m along>.png) and contact_<route>.png, a sheet of them, labelled.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from openpilot.selfdrive.navigation.local_tiles import LocalTiles
from openpilot.selfdrive.navigation.pmtiles import PMTiles
from openpilot.selfdrive.navigation.renderer import MapRenderer
from openpilot.selfdrive.navigation.route import RouteTracker
from openpilot.tools.sim.bridge.gta5.navmodel.frame import lat_lon, tiles_lat0


def route_coordinates(points: np.ndarray, lat0: float = 0.0) -> np.ndarray:
  """navRoute's (lon, lat) for game-metre points."""
  lat, lon = lat_lon(points[:, 0], points[:, 1], lat0)
  return np.column_stack((lon, lat))


DRIVE_STEP = 10.0  # m between the fixes the tracker follows the car by: it searches only -30..+100 m of its progress


def frames_along(renderer: MapRenderer, points: np.ndarray, at: list[float], lat0: float = 0.0):
  """(m along, image, valid) at each of `at` m along the route (ascending), the car driving exactly on the route and
  the tracker following it as mapsd's does."""
  coords = route_coordinates(points, lat0)
  tracker = RouteTracker(coords)
  along = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))))
  out, driven = [], 0.0

  def fix(s):
    lat, lon = lat_lon(np.interp(s, along, points[:, 0]), np.interp(s, along, points[:, 1]), lat0)
    return tracker.project(lon, lat)

  for s in sorted(at):
    while driven + DRIVE_STEP < s:
      driven += DRIVE_STEP
      fix(driven)
    image, valid = renderer.render(*tracker.pose(fix(s)), coords)
    driven = s
    out.append((s, image, valid))
  return out


def contact_sheet(frames: list[tuple[str, np.ndarray]], columns: int = 4):
  from PIL import Image, ImageDraw

  size, label = frames[0][1].shape[0], 14
  rows = (len(frames) + columns - 1) // columns
  sheet = Image.new("L", (columns * (size + 4) + 4, rows * (size + label + 4) + 4), 60)
  draw = ImageDraw.Draw(sheet)
  for k, (text, image) in enumerate(frames):
    x, y = 4 + (k % columns) * (size + 4), 4 + (k // columns) * (size + label + 4)
    draw.text((x, y), text, fill=255)
    sheet.paste(Image.fromarray(image), (x, y + label))
  return sheet


def main():
  from PIL import Image

  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("tiles")
  ap.add_argument("out")
  ap.add_argument("--route", default=str(Path(__file__).parent / "fixtures" / "routes.json") + ":city")
  ap.add_argument("--every", type=float, default=250.0, help="m between frames")
  ap.add_argument("--at", type=float, nargs="*", help="m along the route for frames, in place of --every")
  args = ap.parse_args()
  args.lat0 = tiles_lat0(args.tiles)
  path, name = args.route.rsplit(":", 1)
  points = np.array(json.loads(Path(path).read_text())[name]["points"], dtype=float)
  length = float(np.hypot(*np.diff(points, axis=0).T).sum())
  at = args.at if args.at else list(np.arange(0.0, length, args.every))
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  renderer = MapRenderer(PMTiles(LocalTiles(args.tiles)))
  sheet = []
  for s, image, valid in frames_along(renderer, points, at, args.lat0):
    Image.fromarray(image).save(out / f"frame_{name}_{s:05.0f}.png")
    sheet.append((f"{name} {s:.0f} m{'' if valid else ' INVALID'}", image))
    print(f"{name} {s:7.0f} m valid={valid}")
  contact_sheet(sheet).save(out / f"contact_{name}.png")


if __name__ == "__main__":
  main()
