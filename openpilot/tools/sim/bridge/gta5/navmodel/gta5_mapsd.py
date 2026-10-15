#!/usr/bin/env python3
"""comma's mapsd (selfdrive/navigation/mapsd.py) for the bridge: the nav model's map frames on VisionIPC
(VISION_STREAM_MAP) from the bridge's navRoute (gta5_nav_msgs, GTA5_NAV_MSGS=1) and its simulated GNSS fix (gta5_gnss),
rendered from a local PMTiles file (build_pmtiles.py):

  python -m openpilot.tools.sim.bridge.gta5.navmodel.gta5_mapsd --tiles ~/gta5map/gta5_roads.pmtiles
      [--dump DIR --dump-every 20]

It runs comma's MapsD as it is. The route's and the fix's coordinates are moved from the bridge's frame into the frame
the tiles were built in (their metadata's lat0, frame.py) on their way in (FrameSM); with tiles in the bridge's own
frame (lat0 0) this is just comma's mapsd with NAV_TILES set.
--dump saves every Nth frame as a PNG, with its pose's inputs in frames.jsonl, to look at or to record.
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game
from openpilot.tools.sim.bridge.gta5.navmodel.frame import lat_lon, tiles_lat0

GPS = ("gpsLocation", "gpsLocationExternal")


class FrameSM:
  """A SubMaster as MapsD reads it (updated, valid, recv_time, logMonoTime, and navRoute's coordinates and the GPS
  services' hasFix, speed, latitude and longitude), with the coordinates moved into the frame about lat0."""
  def __init__(self, sm, lat0: float):
    self.sm, self.lat0 = sm, lat0
    self.updated, self.valid, self.recv_time, self.logMonoTime = sm.updated, sm.valid, sm.recv_time, sm.logMonoTime
    self._route = (None, None)  # (logMonoTime, converted navRoute)

  def __getitem__(self, name: str):
    msg = self.sm[name]
    if self.lat0 == 0.0:
      return msg
    if name == "navRoute":
      if self._route[0] != self.sm.logMonoTime[name]:
        coords = msg.coordinates
        lat = np.array([c.latitude for c in coords], dtype=np.float64)
        lon = np.array([c.longitude for c in coords], dtype=np.float64)
        lat, lon = (np.atleast_1d(v).tolist() for v in self.move(lat, lon))
        moved = [SimpleNamespace(latitude=a, longitude=o) for a, o in zip(lat, lon, strict=True)]
        self._route = (self.sm.logMonoTime[name], SimpleNamespace(coordinates=moved))
      return self._route[1]
    if name in GPS:
      lat, lon = self.move(msg.latitude, msg.longitude)
      return SimpleNamespace(hasFix=msg.hasFix, speed=msg.speed, bearingDeg=msg.bearingDeg, latitude=lat, longitude=lon)
    return msg

  def move(self, lat, lon):
    """(lat, lon) in the bridge's frame to the frame about lat0."""
    x, y = to_game(lat, lon)
    return lat_lon(x, y, self.lat0)


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("--tiles", required=True, help="the PMTiles file (build_pmtiles.py)")
  ap.add_argument("--dump", help="save frames here")
  ap.add_argument("--dump-every", type=int, default=20, help="frames between saves (20: one a second)")
  args = ap.parse_args()
  args.lat0 = tiles_lat0(args.tiles)
  print(f"gta5_mapsd: tiles {args.tiles}, frame lat0={args.lat0}", flush=True)

  import openpilot.cereal.messaging as messaging
  import openpilot.selfdrive.navigation.mapsd as mapsd
  from msgq.visionipc import VisionIpcServer
  from openpilot.cereal.visionipc import VisionStreamType
  from openpilot.common.realtime import Ratekeeper
  from openpilot.selfdrive.navigation.renderer import SIZE

  mapsd.LOCAL_TILES = args.tiles
  rk = Ratekeeper(20.0)
  sm = messaging.SubMaster([*GPS, "navRoute"])
  view = FrameSM(sm, args.lat0)
  server = VisionIpcServer("mapsd")
  server.create_buffers_with_sizes(VisionStreamType.VISION_STREAM_MAP, 4, SIZE, SIZE, SIZE * SIZE, SIZE, SIZE * SIZE)
  server.start_listener()
  m = mapsd.MapsD()
  dump = Path(args.dump) if args.dump else None
  log = None
  if dump is not None:
    dump.mkdir(parents=True, exist_ok=True)
    log = (dump / "frames.jsonl").open("a")
  try:
    while True:
      sm.update(0)
      image, timestamp = m.update(view)
      server.send(VisionStreamType.VISION_STREAM_MAP, image.tobytes(), rk.frame, timestamp, time.monotonic_ns())
      if dump is not None and rk.frame % args.dump_every == 0:
        from PIL import Image
        Image.fromarray(image).save(dump / f"map_{rk.frame:07d}.png")
        source = mapsd.latest_gps_source(view)
        fix = view[source] if source is not None else None
        log.write(json.dumps({"frame": rk.frame, "mono": time.monotonic(), "timestamp": timestamp, "lat0": args.lat0,
                              "progress": None if m.tracker is None else float(m.tracker.progress),
                              "fix": None if fix is None else [fix.latitude, fix.longitude, fix.speed]}) + "\n")
        log.flush()
      rk.keep_time()
  finally:
    m.close()
    if log is not None:
      log.close()


if __name__ == "__main__":
  main()
